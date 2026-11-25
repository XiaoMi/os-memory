# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""Public library interface for HyperOS-Mem V11 (DRED + SAGE).

Embed the memory system in your own agent through one class, HyperMemory:

    mem.add(records)                 # save memories: a list (or any iterable) of records
    mem.build()                      # construct the memory graph (DRED + SAGE), once
    mem.search(query)                # find relevant memories by a natural-language query
    mem.read(...)                    # browse memories by event ID, time, entity or anchor

``add`` takes records as in-memory dicts — a list, or a generator for a large
stream; each is validated and persisted as it is read. A record has exactly
three top-level fields, and everything else lives in ``data``::

    {"type": "chat", "timestamp": "2026-05-03T11:46:40+08:00",
     "data": {"from_app": "MyAssistant", "query": "我不吃香菜，帮我找家附近的川菜馆",
              "answer": "为你找到蜀味轩"}}

``timestamp`` is when it happened on the device: integer Unix milliseconds or
an ISO 8601 string (local time without an offset), stored as milliseconds.
Every ``data`` may carry the common fields ``from_app``, ``record_id``
(unique per database), ``anchors`` (``order_id``/``transaction_id``/
``tracking_id``/``reservation_id``/``phone``/``email``/``url``), ``entities``
(``contact``/``merchant``/``product``/``place``/``organizer``) and
``latitude``/``longitude`` (where the device was). Fields by type:

    type         required                          optional
    screen       action_description                typed_text, title, ocr, image_path
    gallery      image_description                 ocr, image_path, captured_at, location
    sms          content                           sender, sender_name
    push         action (receive|open), and title or content
    calendar     title or content                  start_time, end_time, remind_time, calendar_type
    clipboard    content                           action (copy|cut)
    position     latitude+longitude or ssid        location, accuracy
    chat         query                             answer, session_id
    recording    transcript: [{text, offset_ms, speaker}, ...]   audio_path
    note         action (create|edit|delete), and title or content

A bad record raises ``ValueError`` naming its index and rolls the whole batch
back. ``add`` only translates: each record becomes the system's native
OS-trace record (the format the device collector produces), parsed by the
system's own ``OSTraceAdapter``. The full guide is README.md.

HyperOS-Mem builds its event graph in one batch: feed every record with
``add``, call ``build`` once (DRED fast path + SAGE slow path), then
``search`` as often as you like. Building and querying call an
OpenAI-compatible LLM and embedding service, configured once at construction;
``add`` itself is local and makes no network call.

Typical use::

    from hypermem import HyperMemory

    mem = HyperMemory("./user_memory.db", config={
        "llm":       {"base_url": "https://gateway/v1", "model": "qwen2.5-7b",
                       "api_key_env": "OPENAI_API_KEY"},
        "embedding": {"url": "https://gateway/v1/embeddings", "model": "bge-m3",
                       "dimension": 1024, "api_key_env": "OPENAI_API_KEY"},
    })
    mem.add([
        {"type": "screen", "timestamp": 1777770000000,
         "data": {"from_app": "铁路12306", "action_description": "查询 5月8日 北京南→上海虹桥 G15 二等座"}},
        {"type": "sms", "timestamp": "2026-05-03T09:06:00+08:00",
         "data": {"from_app": "短信", "content": "订单E2026050301已支付", "sender": "12306",
                  "anchors": {"order_id": "E2026050301"}}},
    ])
    mem.build()                                   # DRED + SAGE
    hits = mem.search("我最近的订单支付情况？")
    print(hits["context"])                        # prompt-ready text
    mem.close()

The memory system is DRED (online, model-free event attribution) + SAGE (idle
LLM consolidation into events, facts and long-term memory); there are no
alternative modes. See README.md at the repository root for the full guide.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, time, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

from hypermem.adapters import OSTraceAdapter
from hypermem.config import config as formation_config
from hypermem.memory import EventMemory
from hypermem.online.engine import OnlineHyperMem
from hypermem.online.hyperos_retrieval import HyperOSRetriever
from hypermem.online.sage import config as sage_config
from hypermem.online.sage_tools import ToolSAGE
from hypermem.schema import Provenance
from hypermem.schema.serialization import encode_json
from hypermem.standardization import ModelServiceError, make_encoder

__all__ = ["HyperMemory", "ModelServiceError", "format_memories"]

# A build is started once DRED freezes its corpus centre, and built once ToolSAGE
# records a finished run; in between, build() resumes from the saved journals.
_STARTED_KEY = "online_center"
_SAGE_KEY = "sage_last"
_FINISHED = ("complete", "complete_with_skipped_failures")

# -- record specification ---------------------------------------------------
# A record is {"type", "timestamp", "data"}. ``data`` holds the common fields
# below plus the fields of its type. add() only translates: each record becomes
# the OS-trace record the device collector produces (perception_type,
# active_name, from_app and the collector's own data keys), so the system reads
# it exactly as it reads collected data. Facts the system would otherwise not
# see (who spoke, what the user did, a photo's own time and place) are written
# into the observation's text, through the collector's text fields.
_TOP = ("type", "timestamp", "data")
_COMMON = ("from_app", "record_id", "anchors", "entities", "latitude", "longitude")
_ANCHORS = ("order_id", "transaction_id", "tracking_id", "reservation_id", "phone", "email", "url")
# Public entity name -> collector data key.
_ENTITIES = {"contact": "contactName", "merchant": "merchantName", "product": "productName",
             "place": "locationName", "organizer": "organizer"}
# Per type: required fields, at-least-one-of fields, optional fields, and the
# allowed values of ``action`` (None when the type has no action).
_SPEC: dict[str, tuple[tuple, tuple, tuple, tuple | None]] = {
    "screen": (("action_description",), (), ("typed_text", "title", "ocr", "image_path"), None),
    "gallery": (("image_description",), (), ("ocr", "image_path", "captured_at", "location"), None),
    "sms": (("content",), (), ("sender", "sender_name"), None),
    "push": (("action",), ("title", "content"), (), ("receive", "open")),
    "calendar": ((), ("title", "content"), ("start_time", "end_time", "remind_time", "calendar_type"), None),
    "clipboard": (("content",), (), ("action",), ("copy", "cut")),
    "position": ((), (), ("ssid", "location", "accuracy"), None),
    "chat": (("query",), (), ("answer", "session_id"), None),
    "recording": (("transcript",), (), ("audio_path",), None),
    "note": (("action",), ("title", "content"), (), ("create", "edit", "delete")),
}
_STRING_FIELDS = {"action_description", "image_description", "title", "ocr", "image_path", "location",
                  "content", "sender", "sender_name", "action", "ssid", "query", "answer", "session_id",
                  "audio_path", "from_app", "record_id", "typed_text", "calendar_type"}
_TIME_FIELDS = ("captured_at", "start_time", "end_time", "remind_time")
# Calendar times and type the collector has no keys for, written into the text in this order.
_CALENDAR_TEXT = (("start_time", "开始时间"), ("end_time", "结束时间"), ("remind_time", "提醒时间"))
# Collector action labels (active_name), as the device collector writes them.
_ACTIVE_NAME = {
    ("sms", None): "接收短信",
    ("push", "receive"): "PUSH接收", ("push", "open"): "PUSH打开",
    ("clipboard", "copy"): "复制", ("clipboard", "cut"): "剪切", ("clipboard", None): "剪贴板新增",
    ("calendar", None): "日历新增", ("gallery", None): "相册新增",
    ("note", "create"): "笔记新增", ("note", "edit"): "笔记编辑", ("note", "delete"): "笔记删除",
}
# What the user did, written into the text (the system reads active_name only
# when a record has no text at all).
_ACTION_TEXT = {
    ("push", "open"): "用户打开了这条通知",
    ("clipboard", "copy"): "用户复制了这段内容", ("clipboard", "cut"): "用户剪切了这段内容",
    ("note", "create"): "用户新建了这条笔记", ("note", "edit"): "用户编辑了这条笔记",
    ("note", "delete"): "用户删除了这条笔记",
}


def _nonempty(value: Any) -> bool:
    return value is not None and (not isinstance(value, str) or value.strip() != "")


def _is_int_ms(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# 1e11 ms is March 1973; a smaller integer record time is almost surely Unix seconds.
_MIN_MS = 100_000_000_000


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _text(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _record_ms(value: Any, name: str = "timestamp") -> int:
    """A record time as Unix milliseconds: an integer as is, or an ISO 8601 string."""
    if _is_int_ms(value):
        if value < _MIN_MS:
            raise ValueError(f"{name} must be integer Unix milliseconds; {value} looks like Unix seconds "
                             f"(before 1973 in milliseconds), multiply it by 1000")
        return value
    if isinstance(value, str):
        try:
            return _parse_moment(value, name)[0]
        except ValueError:
            pass
    raise ValueError(f"{name} must be integer Unix milliseconds or an ISO 8601 date/time "
                     f"such as '2026-05-03T08:30:00+08:00', got {value!r}")


def _shown_time(value: Any, ms: int) -> str:
    """A time for the text: as the caller wrote it, or local time for milliseconds."""
    if isinstance(value, str):
        return value.strip()
    return datetime.fromtimestamp(ms / 1000).astimezone().isoformat(timespec="minutes")


def _translate(record: Any) -> tuple[dict, list[tuple[str, int, dict]]]:
    """Validate one record; return (record to store, [(perception_type, ts_ms, event), ...]).

    The stored record is the caller's record with ``timestamp`` in Unix
    milliseconds. Each event is a collector OS-trace event. One record yields
    one observation, except a chat turn with an answer and a position with
    both coordinates and ssid (two each) and a recording (one per segment).
    Raises ValueError on any invalid field.
    """
    if not isinstance(record, dict):
        raise ValueError("each record must be an object")
    unknown = sorted(set(record) - set(_TOP))
    if unknown:
        raise ValueError(f"unknown field(s) {', '.join(unknown)}; a record has only type, timestamp "
                         "and data (from_app, anchors and the rest go inside data)")
    kind = record.get("type")
    if kind not in _SPEC:
        raise ValueError(f"unknown type {kind!r}; one of: {', '.join(_SPEC)}")
    if "timestamp" not in record:
        raise ValueError("record requires 'timestamp'")
    ts = _record_ms(record["timestamp"])
    data = record.get("data")
    if not isinstance(data, dict):
        raise ValueError("record requires a 'data' object")
    required, any_of, _, actions = _SPEC[kind]
    for field in required:
        if field not in data or not _nonempty(data[field]):
            raise ValueError(f"type {kind!r} requires data.{field}")
    if any_of and not any(_nonempty(data.get(f)) for f in any_of):
        raise ValueError(f"type {kind!r} requires data.{' or data.'.join(any_of)}")
    for field in _STRING_FIELDS & set(data):
        if data[field] is not None and not _text(data[field]):
            raise ValueError(f"data.{field} must be a non-empty string")
    action = data.get("action")
    if actions is not None and action is not None and action not in actions:
        raise ValueError(f"data.action must be one of {', '.join(actions)}, got {action!r}")
    times = {f: _record_ms(data[f], f"data.{f}") for f in _TIME_FIELDS if data.get(f) is not None}
    common = _common(data)

    if kind == "screen":
        # typed_text is what the input field holds after typing: the collector's text_after.
        events = [_event("ScreenPerception", None, {"vlm_caption": data["action_description"],
                                                    **_rename(data, {"typed_text": "text_after"}),
                                                    **_pick(data, ("title", "ocr", "image_path"))})]
    elif kind == "gallery":
        said = []
        if "captured_at" in times:
            said.append("拍摄时间：" + _shown_time(data["captured_at"], times["captured_at"]))
        if _text(data.get("location")):
            said.append("拍摄地点：" + data["location"].strip())
        events = [_event("GalleryPerception", _ACTIVE_NAME[(kind, None)],
                         {"vlm_caption": data["image_description"], **_pick(data, ("ocr", "image_path")),
                          **({"description": "；".join(said)} if said else {}),
                          **({"captured_at": times["captured_at"]} if "captured_at" in times else {}),
                          **_pick(data, ("location",))})]
    elif kind == "sms":
        events = [_event("SmsPerception", _ACTIVE_NAME[(kind, None)],
                         {"content": data["content"], **_rename(data, {"sender": "phoneNum",
                                                                        "sender_name": "contactName"})})]
    elif kind == "push":
        events = [_event("PushPerception", _ACTIVE_NAME[(kind, action)],
                         {**_pick(data, ("title", "content")), **_said(kind, action)})]
    elif kind == "calendar":
        if "start_time" in times and "end_time" in times and times["end_time"] < times["start_time"]:
            raise ValueError("data.end_time must not be earlier than data.start_time")
        said = [f"{label}：" + _shown_time(data[f], times[f]) for f, label in _CALENDAR_TEXT if f in times]
        if _text(data.get("calendar_type")):
            said.append("日历类型：" + data["calendar_type"].strip())
        events = [_event("CalendarPerception", _ACTIVE_NAME[(kind, None)],
                         {**_pick(data, ("title", "content")),
                          **({"description": "；".join(said)} if said else {}),
                          **{f: times[f] for f, _ in _CALENDAR_TEXT if f in times},
                          **_pick(data, ("calendar_type",))})]
    elif kind == "clipboard":
        events = [_event("ClipboardPerception", _ACTIVE_NAME[(kind, action)],
                         {"content": data["content"], **_said(kind, action)})]
    elif kind == "position":
        events = _position_events(data, common)
    elif kind == "chat":
        events = _chat_events(data)
    elif kind == "recording":
        events = _recording_events(data)
    else:  # note
        # Notes are the user's own writing; edits and new notes speak for the user.
        events = [_event("NotePerception", _ACTIVE_NAME[(kind, action)],
                         {**_pick(data, ("title", "content")), **_said(kind, action),
                          **({"speaker_role": "user"} if action in ("create", "edit") else {})})]

    out = []
    for offset, (perception_type, active_name, unit) in events:
        for key, value in common.items():
            unit.setdefault(key, value)  # a field the type already set (e.g. sms sender_name) wins
        event: dict[str, Any] = {"perception_type": perception_type, "data": unit}
        if active_name:
            event["active_name"] = active_name
        if "from_app" in data:
            event["from_app"] = data["from_app"]
        out.append((perception_type, ts + offset, event))
    return {**record, "timestamp": ts}, out


def _event(perception_type: str, active_name: str | None, unit: dict, offset: int = 0):
    return offset, (perception_type, active_name, unit)


def _pick(data: dict, fields: tuple) -> dict:
    return {f: data[f] for f in fields if _nonempty(data.get(f))}


def _rename(data: dict, names: dict) -> dict:
    return {names[f]: data[f] for f in names if _nonempty(data.get(f))}


def _said(kind: str, action: str | None) -> dict:
    text = _ACTION_TEXT.get((kind, action))
    return {"description": text} if text else {}


def _common(data: dict) -> dict[str, Any]:
    """Validated anchors, entities and device position, as collector data keys."""
    out: dict[str, Any] = {}
    for name, allowed in (("anchors", {a: a for a in _ANCHORS}), ("entities", _ENTITIES)):
        values = data.get(name)
        if values is None:
            continue
        if not isinstance(values, dict):
            raise ValueError(f"data.{name} must be an object")
        unknown = sorted(set(values) - set(allowed))
        if unknown:
            raise ValueError(f"unknown data.{name} key(s) {', '.join(unknown)}; one of: {', '.join(allowed)}")
        for key, value in values.items():
            if not _text(value):
                raise ValueError(f"data.{name}.{key} must be a non-empty string")
            out[allowed[key]] = value
    lat, lon = data.get("latitude"), data.get("longitude")
    if lat is not None or lon is not None:
        if not (_is_num(lat) and _is_num(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("data.latitude and data.longitude must be given together as numbers "
                             "within -90..90 and -180..180")
        out["latitude"], out["longitude"] = lat, lon
    return out


def _position_events(data: dict, common: dict) -> list:
    events = []
    if "latitude" in common:
        lat, lon = common["latitude"], common["longitude"]
        unit: dict[str, Any] = {}
        if data.get("accuracy") is not None:
            if not _is_num(data["accuracy"]) or data["accuracy"] < 0:
                raise ValueError("data.accuracy must be a non-negative number (metres)")
            unit["accuracy"] = data["accuracy"]
        place = data.get("location")
        unit["content"] = f"位置：{place.strip()}" if _text(place) else f"位置：纬度 {lat}，经度 {lon}"
        events.append(_event("GPSPerception", None, unit))
    if _text(data.get("ssid")):
        ssid = data["ssid"]
        events.append(_event("WifiPerception", None, {"connectedWifi": ssid, "content": f"连接 WiFi：{ssid}"},
                             offset=len(events)))
    if not events:
        raise ValueError("type 'position' requires data.latitude and data.longitude, or data.ssid")
    return events


def _chat_events(data: dict) -> list:
    # The speaker is written into the text: the system reads text, not roles.
    # The answer is placed 1 ms after the query so the build keeps the turn in order.
    session = {"session_id": data["session_id"]} if _text(data.get("session_id")) else {}
    events = [_event("AppPerception", None,
                     {"content": "用户：" + data["query"], "speaker_role": "user", **session})]
    if _nonempty(data.get("answer")):
        events.append(_event("AppPerception", None,
                             {"content": "助手：" + data["answer"], "speaker_role": "assistant", **session},
                             offset=1))
    return events


def _recording_events(data: dict) -> list:
    transcript = data["transcript"]
    if not isinstance(transcript, list) or not transcript:
        raise ValueError("data.transcript must be a non-empty list of {text, offset_ms, speaker}")
    events = []
    for i, part in enumerate(transcript):
        where = f"data.transcript[{i}]"
        if not isinstance(part, dict) or set(part) != {"text", "offset_ms", "speaker"}:
            raise ValueError(f"{where} must be an object with exactly text, offset_ms and speaker")
        if not _text(part["text"]):
            raise ValueError(f"{where}.text must be a non-empty string")
        if not _is_int_ms(part["offset_ms"]) or part["offset_ms"] < 0:
            raise ValueError(f"{where}.offset_ms must be a non-negative integer")
        speaker = part["speaker"]
        if speaker is not None and not _text(speaker):
            raise ValueError(f"{where}.speaker must be a non-empty string or null")
        unit: dict[str, Any] = {}
        if speaker == "user":
            unit.update(transcript="用户：" + part["text"], speaker_role="user")
        elif speaker is not None:
            unit.update(transcript=f"{speaker}：" + part["text"], contactName=speaker)
        else:
            unit["transcript"] = part["text"]
        if _text(data.get("audio_path")):
            unit["audio_path"] = data["audio_path"]
        events.append(_event("AudioPerception", None, unit, offset=part["offset_ms"]))
    return events


class HyperMemory:
    """A HyperOS-Mem memory library backed by a single SQLite file.

    One instance owns one memory database for one user. The database is built
    once from the texts added to it; to memorise a further batch, open a new
    database. All methods are synchronous and single-threaded.
    """

    def __init__(self, db_path: str | Path = ":memory:", *, config: dict | None = None,
                 source_id: str | None = None) -> None:
        """Open or create a memory database.

        db_path    SQLite file path, or ":memory:" for an ephemeral database.
        config     LLM and embedding services (see module docstring). Required
                   for ``build`` and ``search``; ``add`` and ``read`` work
                   without it.
        source_id  Stable namespace for observation IDs; defaults to the path.
        """
        self.db_path = str(db_path)
        self._source_id = source_id or f"hypermem-api:{self.db_path}"
        self.memory = EventMemory(self.db_path)
        self._formation, self._sage, self._llm_missing = _resolve_config(config) if config else (None, None, [])
        # Two counters, both continuing past whatever is already stored so IDs
        # stay stable across sessions and never collide with an earlier add:
        #   _record -> one per input record (becomes the raw_records row / raw_id)
        #   _obs    -> one per observation (becomes the observation_id)
        # A segmented recording yields several observations under one raw record.
        self._record = self.memory.counts()["raw_records"]
        self._obs = self.memory.counts()["observations"]

    # -- ingest -----------------------------------------------------------

    def add(self, records: Iterable[dict], *, user: str | None = None) -> dict[str, int]:
        """Save memories: validate and persist a batch of records. Local only;
        makes no network call.

        ``records`` is any iterable of record dicts — a list, or a generator for
        a large stream; records are persisted as they are read and never
        collected in memory. The record format is in the module docstring.
        A record that fails validation raises ``ValueError`` naming its index
        (``records[12]: ...``) and the whole batch is rolled back. ``user`` sets
        the owner (one database holds a single user). Call order does not
        matter — the build sorts by time.

        Returns ``{"records": n, "observations": m}``. A chat turn with both a
        query and an answer, or a recording with segments, yields more than one
        observation.
        """
        if isinstance(records, (str, bytes, Path, dict)):
            raise TypeError("add takes a list (or iterable) of records; to load a JSONL file: "
                            "add(json.loads(line) for line in open(path, encoding='utf-8') if line.strip())")
        return self._add_labelled(((f"records[{i}]", r) for i, r in enumerate(records)), user)

    def _add_labelled(self, labelled: Iterable[tuple[str, Any]], user: str | None) -> dict[str, int]:
        """Persist (label, record) pairs in one transaction; errors name the label."""
        if self.is_started:
            raise RuntimeError(
                "This memory is already built or being built; OS-Memory builds in one batch. "
                "Open a new HyperMemory database to memorise additional records.")
        counts = {"records": 0, "observations": 0}
        # One transaction wraps the whole stream: records land on disk (SQLite)
        # as they are read, and a bad record rolls the batch back cleanly.
        # Reentrant savepoints let each per-record add_observation transaction
        # nest safely inside this one.
        with self.memory.transaction():
            for label, record in labelled:
                try:
                    written = self._ingest_record(record, user)
                except ValueError as exc:
                    raise ValueError(f"{label}: {exc}") from exc
                counts["records"] += 1
                counts["observations"] += written
        return counts

    def _ingest_record(self, record: Any, user: str | None) -> int:
        """Validate one record and persist its observation(s); returns how many."""
        stored, units = _translate(record)  # validates; raises ValueError on bad input
        record_id = stored["data"].get("record_id")
        # One raw record per input record, shared by all its observations. A
        # caller's record_id names it, so a repeated record_id is a primary-key hit.
        self._record += 1
        key = ["record_id", record_id] if record_id else ["record", self._record]
        raw_id = "R-" + hashlib.sha256(encode_json([self._source_id, *key]).encode("utf-8")).hexdigest()
        if record_id:
            try:
                self.memory.get_raw_record(raw_id)
            except KeyError:
                pass
            else:
                raise ValueError(f"duplicate record_id {record_id!r}")
        record_sha256 = hashlib.sha256(encode_json(stored).encode("utf-8")).hexdigest()
        for index, (perception_type, ts, event) in enumerate(units):
            self._obs += 1
            obs_id = "O-" + hashlib.sha256(
                encode_json([self._source_id, "obs", self._obs]).encode("utf-8")).hexdigest()
            os_trace = {"ts": ts, "type": "real", "payload": {"event": event}}
            metadata: dict[str, Any] = {
                "source_id": self._source_id,
                "record_number": self._record,
                "segment_index": index,
                "record_sha256": record_sha256,
                "api_type": stored["type"],
                "input_format": "type_timestamp_data_record",
            }
            if record_id:
                metadata["record_id"] = record_id
            if user is not None:
                metadata["user_id"] = user
            # OSTraceAdapter does the parsing, anchor extraction and validation;
            # the observation's provenance points to the shared raw_id, and
            # add_observation persists the caller's record under it — raw_records
            # holds what was submitted (timestamp in milliseconds), not the
            # internal os_trace translation. The raw insert is idempotent, so
            # repeating it for each observation stores the record just once.
            observation = OSTraceAdapter().parse(
                os_trace, observation_id=obs_id,
                provenance=Provenance(raw_id, None, perception_type, metadata))
            type(observation).from_json(observation.to_json())  # validate both directions
            self.memory.add_observation(observation, stored)
        return len(units)

    # -- build ------------------------------------------------------------

    def build(self, *, progress=None) -> dict:
        """Construct the memory graph: DRED fast path, then the SAGE slow path.

        Calls the configured LLM and embedding services. Runs once per database;
        returns a short report. Raises if no observations were added, if no
        config was supplied, or if the database is already built.

        If the LLM or embedding service fails (unreachable, timing out,
        overloaded, or refusing the key, URL or model), the build stops with
        ``ModelServiceError`` and no task spends its retry budget on the
        outage. Progress is saved: once the service is restored, call
        ``build()`` again and it resumes where it stopped (the report then has
        ``"resumed": True``). A build interrupted any other way resumes too.

        ``progress``, if given, is called as ``progress(info)`` with a dict whose
        ``stage`` is ``"dred"``, ``"sage_boundary"`` or ``"sage_event_tools"``,
        plus that stage's counters.
        """
        if self._formation is None:
            raise RuntimeError("build requires an LLM/embedding config; pass one to HyperMemory(...)")
        if self._llm_missing:
            raise ValueError(f"config is missing {', '.join(self._llm_missing)}; build needs an LLM service")
        if self.is_built:
            raise RuntimeError("This memory is already built")
        if self.memory.counts()["observations"] == 0:
            raise RuntimeError("Add at least one memory with add before build")
        resumed = self.is_started
        dred_progress, boundary_progress, tools_progress = _progress_callbacks(progress)
        # Every step skips work its journal records as done, so the same three
        # steps both build and resume.
        try:
            # DRED: online, model-free attribution of every observation to an
            # event, which also freezes the corpus centre used by retrieval.
            engine = OnlineHyperMem(self.memory, self._formation)
            dred_report = engine.replay(progress=dred_progress, detailed_progress=False)
            # SAGE step 1: idle boundary consolidation of the DRED events.
            engine.run_slow(is_idle=True, progress=boundary_progress)
            del engine
            # SAGE step 2: ToolSAGE (SplitEvent, SummarizeEvent, ExtractFacts) and
            # LongMemory consolidation, writing the retrievable event graph.
            tools = ToolSAGE(self.memory, self._sage, progress=tools_progress)
            sage_report = tools.run()
        except ModelServiceError as exc:
            raise ModelServiceError(
                f"{exc}. The build is paused with its progress saved; restore the service "
                "and call build() again to continue.") from exc
        if sage_report.get("status") not in _FINISHED:
            raise RuntimeError(
                f"SAGE completion checks failed (status {sage_report.get('status')!r}); "
                "the memory is not built")
        return {"status": "built", "resumed": resumed, "observations": self.memory.counts()["observations"],
                "events": len(self.memory.list_events()),
                "dred": {"events": dred_report.get("events")},
                "sage": {"status": sage_report.get("status")}}

    # -- browse --------------------------------------------------------------

    def read(
        self,
        *,
        event_id: str | None = None,
        start: str | datetime | date | int | None = None,
        end: str | datetime | date | int | None = None,
        entity: str | None = None,
        anchor: str | None = None,
        limit: int = 20,
        include_facts: bool = True,
        include_observations: bool = False,
        include_raw: bool = False,
        include_long_memories: bool = False,
    ) -> dict:
        """Browse the memory graph by event ID, time window, entity, or anchor.

        Memory layer architecture::

            长期记忆 (long_memory)         <- 跨多个事件的总结
                └─ 事件 (event)            <- 一次完整的事（查票+付款+出行）
                     ├─ 事实 (fact)        <- SAGE 从事件里抽的一句话结论
                     └─ 观测 (observation) <- 解析后的单条记录（文本/anchor/时间）
                          └─ 原始记录 (raw) <- 你 add 提交的那条记录

        Unlike ``search``, this method does not call any model or embedding
        service and works on a database that has not yet been built (SAGE-produced
        fields such as ``title``, ``summary``, and ``facts`` are simply ``None``
        before ``build()``).

        Parameters
        ----------
        event_id
            Return exactly one event by its ID; all other filters are ignored.
        start
            Keep only events that start at or after this time: a date or time
            string such as ``"2026-05-03"`` or ``"2026-05-03 08:30"`` (ISO 8601,
            an offset like ``"+08:00"`` optional), or a ``datetime``/``date``.
            A time without an offset is local time.
        end
            Keep only events that start at or before this time, written like
            ``start``. A date alone includes that whole day, so
            ``start="2026-05-03", end="2026-05-03"`` selects that one day.
        entity
            Keep only events that include this string in their entities list
            (case-insensitive substring match).
        anchor
            Keep only events whose anchors dict contains this value anywhere
            (e.g. ``"E2026050301"`` to find everything linked to that order ID).
        limit
            Maximum number of events to return (default 20).
        include_facts
            Include SAGE-distilled facts for each event (default True).
            Each fact is a one-sentence conclusion with a list of supporting
            observation IDs. Only populated after ``build()``.
        include_observations
            Include the list of parsed observations for each event (default False).
            Each observation carries the text the system extracted, its anchors,
            its source (app name), and its timestamp.
        include_raw
            Include the original records you submitted to ``add``,
            de-duplicated by ``raw_id`` across the event's observations
            (default False). These are the dicts you passed, with
            ``timestamp`` in milliseconds — the deepest level of provenance.
        include_long_memories
            Include cross-event long-term memories that overlap the returned
            events (default False). Only populated after ``build()``.

        Returns a dict with keys:

        - ``events`` — list of event dicts, each containing ``event_id``,
          ``time_span``, ``state``, ``entities``, ``anchors``, ``confidence``,
          ``title``, ``summary``, and optionally ``facts``, ``observations``,
          ``raw_records``.
        - ``long_memories`` — list of long-memory dicts (always present, empty
          when ``include_long_memories=False``).
        - ``total`` — total matching events before the ``limit`` is applied.
        """
        # --- collect candidate events ---
        if event_id is not None:
            try:
                candidates = [self.memory.get_event(event_id)]
            except KeyError:
                return {"events": [], "long_memories": [], "total": 0}
        else:
            start, end = _time_bound(start, "start"), _time_bound(end, "end", end=True)
            candidates = []
            for ev in self.memory.iter_events(include_embedding=False):
                ts_ms = int(ev.start_time.timestamp() * 1000)
                if start is not None and ts_ms < start:
                    continue
                if end is not None and ts_ms > end:
                    continue
                if entity is not None and not any(
                    entity.lower() in e.lower() for e in ev.entities
                ):
                    continue
                if anchor is not None and not any(
                    anchor in vals for vals in ev.anchors.values()
                ):
                    continue
                candidates.append(ev)

        total = len(candidates)
        candidates = candidates[:limit]

        # --- SAGE helpers: both tables may be absent before build() ---
        def _sage_event_record(eid: str) -> dict:
            try:
                from hypermem.online.sage import row as sage_row
                return sage_row(self.memory, "sage_events", eid) or {}
            except Exception:
                return {}

        def _sage_facts_for(eid: str) -> list:
            try:
                from hypermem.online.sage import rows as sage_rows
                facts = sage_rows(
                    self.memory, "sage_facts",
                    where="json_extract(record_json,'$.event_id')=?",
                    params=(eid,),
                )
                return [{"fact_id": fid, "text": f.get("text"),
                         "support": f.get("support", [])}
                        for fid, f in facts.items()]
            except Exception:
                return []

        # --- build per-event entries ---
        events: list[dict] = []
        matched_event_ids: set[str] = set()

        for ev in candidates:
            sage_ev = _sage_event_record(ev.event_id)
            entry: dict[str, Any] = {
                "event_id": ev.event_id,
                "time_span": [ev.start_time.isoformat(), ev.end_time.isoformat()],
                "state": ev.state,
                "entities": ev.entities,
                "anchors": ev.anchors,
                "confidence": ev.confidence,
                "title": sage_ev.get("title"),
                "summary": sage_ev.get("summary"),
            }
            if include_facts:
                entry["facts"] = _sage_facts_for(ev.event_id)
            if include_observations or include_raw:
                obs_list = self.memory.get_observations(ev.event_id)
                if include_observations:
                    entry["observations"] = []
                    for o in obs_list:
                        ts = o.timestamp
                        ts_iso = (ts.start if hasattr(ts, "start") else ts).isoformat()
                        entry["observations"].append({
                            "observation_id": o.observation_id,
                            "timestamp": ts_iso,
                            "source": o.source,
                            "anchors": o.anchors,
                        })
                if include_raw:
                    seen: dict[str, Any] = {}
                    for o in obs_list:
                        rid = o.provenance.raw_id
                        if rid not in seen:
                            try:
                                seen[rid] = self.memory.get_raw_record(rid)
                            except KeyError:
                                pass
                    entry["raw_records"] = [
                        {"raw_id": rid, "record": rec}
                        for rid, rec in seen.items()
                    ]
            events.append(entry)
            matched_event_ids.add(ev.event_id)

        # --- long memories that overlap the matched events ---
        long_mems: list[dict] = []
        if include_long_memories and matched_event_ids:
            try:
                from hypermem.online.sage import rows as sage_rows
                all_lm = sage_rows(self.memory, "sage_long_memory")
                for mid, lm in all_lm.items():
                    support = set(lm.get("support_events") or [])
                    if support & matched_event_ids:
                        long_mems.append({
                            "memory_id": mid,
                            "text": lm.get("text"),
                            "confidence": lm.get("confidence"),
                            "support_events": lm.get("support_events", []),
                            "evidence_ids": lm.get("evidence_ids", []),
                        })
            except Exception:
                pass

        return {"events": events, "long_memories": long_mems, "total": total}

    # -- query ------------------------------------------------------------

    def search(self, query: str, *, top_k: int = 20, as_text: bool = True) -> dict:
        """Retrieve memories relevant to a natural-language query.

        Returns a dict with ``events`` (each with title, summary, time span and
        grounded facts), ``long_memories``, and ``scores``. When ``as_text`` is
        true (default) a prompt-ready ``context`` string is included too.
        Requires a built database and an embedding config.
        """
        if self._sage is None:
            raise RuntimeError("search requires an LLM/embedding config; pass one to HyperMemory(...)")
        if not self.is_built:
            if self.is_started:
                raise RuntimeError("Call build() before search(); the last build did not finish, "
                                   "and build() resumes it")
            raise RuntimeError("Call build() before search(); the memory graph is not constructed yet")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("search requires a non-empty query")
        if not _is_int_ms(top_k) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        retriever = HyperOSRetriever(SimpleNamespace(
            m=self.memory, cfg=self._sage, encoder=make_encoder(self._sage["embedding"]),
            center=self.memory.get_metadata(_STARTED_KEY),
            events={event.event_id: event for event in self.memory.iter_events()}))
        retrieved = retriever.retrieve(query)
        result = _shape_result(query, retrieved, top_k=top_k)
        if as_text:
            result["context"] = format_memories(result)
        return result

    # -- lifecycle --------------------------------------------------------

    @property
    def is_built(self) -> bool:
        """Whether the memory graph has been constructed (build has finished)."""
        return (self.memory.get_metadata(_SAGE_KEY) or {}).get("status") in _FINISHED

    @property
    def is_started(self) -> bool:
        """Whether a build has started; records can no longer be added then."""
        return self.memory.get_metadata(_STARTED_KEY) is not None

    def count(self) -> dict[str, int]:
        """Stored observation/event/relation counts."""
        return self.memory.counts()

    def close(self) -> None:
        """Close the database connection."""
        self.memory.close()

    def __enter__(self) -> "HyperMemory":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# -- helpers --------------------------------------------------------------

def _parse_moment(value: Any, name: str) -> tuple[int, bool]:
    """(Unix milliseconds, whether only a date was given) for an ISO 8601 string,
    ``datetime`` or ``date``. A time without an offset is local time."""
    whole_day = False
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, date):
        moment, whole_day = datetime.combine(value, time()), True
    elif isinstance(value, str):
        text = value.strip()
        # Python 3.10's fromisoformat does not read a trailing "Z".
        iso = text[:-1] + "+00:00" if text[-1:] in ("Z", "z") else text
        try:
            moment = datetime.fromisoformat(iso)
        except ValueError:
            raise ValueError(f"{name} must be a date or time such as '2026-05-03' or "
                             f"'2026-05-03 08:30', got {value!r}") from None
        whole_day = re.fullmatch(r"\d{4}-\d{2}-\d{2}", text) is not None
    else:
        raise ValueError(f"{name} must be a date or time such as '2026-05-03' or "
                         f"'2026-05-03 08:30', got {value!r}")
    if moment.tzinfo is None:
        moment = moment.astimezone()  # interpret as local time
    return int(moment.timestamp() * 1000), whole_day


def _time_bound(value: Any, name: str, *, end: bool = False) -> int | None:
    """A read() time bound as Unix milliseconds.

    Accepts an ISO 8601 date or time string, a ``datetime`` or ``date``, or
    (as before) integer Unix milliseconds. A time without an offset is local
    time. A date alone, given as ``end``, includes that whole day.
    """
    if value is None or _is_int_ms(value):
        return value
    ms, whole_day = _parse_moment(value, name)
    return ms + 86_400_000 - 1 if end and whole_day else ms

def _resolve_config(config: dict) -> tuple[dict, dict, list[str]]:
    """Translate a user config into validated formation and SAGE settings.

    Mirrors the protocol wiring used by the benchmark: the embedding URL is an
    OpenAI-compatible endpoint ending in /v1/embeddings; the LLM base URL ends
    in /v1. Secrets stay in environment variables named by ``api_key_env``.

    Only ``build`` calls the LLM, so the ``llm`` section may be absent or
    incomplete; the third value lists its missing keys and ``build`` rejects
    the config when it is not empty. ``search`` needs the embedding section only.
    """
    if "embedding" not in config:
        raise ValueError("config needs an 'embedding' section")
    emb_in, llm_in = config["embedding"], config.get("llm") or {}
    if not isinstance(llm_in, dict):
        raise ValueError("config 'llm' must be an object")
    llm_missing = [f"llm.{k}" for k in ("base_url", "model") if not _nonempty(llm_in.get(k))]
    missing = []
    if not _nonempty(emb_in.get("url")) and not _nonempty(emb_in.get("base_url")):
        missing.append("embedding.url")
    missing += [f"embedding.{k}" for k in ("model", "dimension") if not _nonempty(emb_in.get(k))]
    if missing:
        raise ValueError(f"config is missing {', '.join(missing)}")

    emb_url = emb_in.get("url")
    if emb_url:
        if "/v1/" not in emb_url:
            raise ValueError("embedding.url must be an OpenAI-compatible URL containing /v1/ (e.g. .../v1/embeddings)")
        base, tail = emb_url.rsplit("/v1/", 1)
        emb_base, emb_path = base, "/v1/" + tail
    else:
        emb_base, emb_path = emb_in["base_url"].rstrip("/"), emb_in.get("path", "/v1/embeddings")
    embedding = {"backend": "http", "base_url": emb_base, "path": emb_path,
                 "model": emb_in["model"], "dimension": int(emb_in["dimension"]),
                 "batch_size": int(emb_in.get("batch_size", 32)), "protocol": emb_in.get("protocol", "openai"),
                 "timeout": int(emb_in.get("timeout", 60)), "retries": int(emb_in.get("retries", 2)),
                 "api_key_env": emb_in.get("api_key_env", "")}

    llm = {"base_url": str(llm_in.get("base_url") or "").rstrip("/"), "path": llm_in.get("path", "/chat/completions"),
           "model": llm_in.get("model") or "", "temperature": 0, "max_tokens": int(llm_in.get("max_tokens", 8192)),
           "timeout": int(llm_in.get("timeout", 180)), "retries": int(llm_in.get("retries", 2)),
           "api_key_env": llm_in.get("api_key_env", "")}

    formation = formation_config({"llm": llm, "embedding": embedding,
                                  "online": {"consolidation": {"batch_size": 2, "concurrency": 8}}})
    sage = sage_config({"llm": dict(llm), "embedding": dict(embedding), "event": {"concurrency": 8}})
    return formation, sage, llm_missing


def _progress_callbacks(progress):
    """Adapt one ``progress(info: dict)`` callback to the three build stages,
    each of which reports in its own calling convention."""
    if progress is None:
        return None, None, None

    def dred(count, report):
        progress({"stage": "dred", **report, "processed": count})

    def boundary(**info):
        progress({"stage": "sage_boundary", **info})

    def tools(info):
        progress(dict(info))

    return dred, boundary, tools


def _shape_result(query: str, retrieved: dict, *, top_k: int) -> dict:
    """Reshape the retriever output into a stable, caller-friendly structure."""
    scores = retrieved["node_scores"]
    # The retriever lists events by start time; keep the top_k most relevant,
    # then return them in that time order.
    ranked = sorted(retrieved["events"], key=lambda e: -scores.get(e["event_id"], float("-inf")))
    kept = {event["event_id"] for event in ranked[:top_k]}
    events = []
    for event in (e for e in retrieved["events"] if e["event_id"] in kept):
        eid = event["event_id"]
        events.append({
            "event_id": eid, "score": scores.get(eid),
            "title": event.get("title"), "summary": event.get("summary"),
            "time_span": event.get("span"),
            "facts": [{"text": fact.get("text"), "support": fact.get("support", [])}
                      for fact in event.get("facts", [])],
        })
    long_memories = [{"memory_id": item.get("memory_id"), "score": scores.get(item.get("memory_id")),
                      "text": item.get("text"), "evidence": item.get("evidence_ids", [])}
                     for item in retrieved.get("long_memories", [])]
    return {"query": query, "events": events, "long_memories": long_memories, "scores": scores}


def format_memories(result: dict) -> str:
    """Render a query result as prompt-ready text for an agent's context."""
    lines: list[str] = []
    for memory in result.get("long_memories", []):
        if memory.get("text"):
            lines.append(f"[长期记忆] {memory['text']}")
    for event in result.get("events", []):
        header = event.get("title") or event.get("event_id")
        span = event.get("time_span")
        when = f"（{span[0]} ~ {span[1]}）" if isinstance(span, (list, tuple)) and len(span) == 2 else ""
        lines.append(f"[事件] {header}{when}")
        if event.get("summary"):
            lines.append(f"  摘要：{event['summary']}")
        for fact in event.get("facts", []):
            if fact.get("text"):
                lines.append(f"  事实：{fact['text']}")
    return "\n".join(lines)
