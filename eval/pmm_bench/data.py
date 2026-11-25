# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""PMM-Bench data access: users, questions and observation streams.

Construction reads observations only. Reference answers, evidence and
gold records are used for scoring and never reach a memory system.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
from pathlib import Path
import re

from hypereval.protocol import MemoryEvent, Question

CATEGORIES = ("Single", "Cross", "Temp", "Track", "Habit")
CATEGORY_ALIASES = {
    "事实回忆": "Single", "单跳": "Single", "Single": "Single",
    "汇总统计": "Cross", "Cross": "Cross",
    "时序承接": "Temp", "Temp": "Temp",
    "事件经过": "Track", "Track": "Track",
    "习惯流程": "Habit", "Habit": "Habit",
}
ABSTAIN = "证据不足，无法回答"
TZ = timezone(timedelta(hours=8))
INPUTS = {"cleaned": "data/cleaned_observations.jsonl", "raw": "observations.jsonl"}


def clean_text(value) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", str(value))).replace("**", "").strip()


def file_hash(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def list_users(root) -> list[str]:
    """Every directory under users/ that has a qa.json, sorted by name."""
    base = Path(root) / "users"
    if not base.is_dir():
        raise FileNotFoundError(f"No users/ directory under {root}")
    return sorted(p.name for p in base.iterdir() if p.is_dir() and (p / "qa.json").is_file())


def select_users(root, spec: str | None) -> list[str]:
    available = list_users(root)
    if not spec or spec == "all":
        return available
    wanted = [x.strip() for x in spec.split(",") if x.strip()]
    missing = [u for u in wanted if u not in available]
    if missing:
        raise ValueError(f"Unknown users: {missing}. Available: {available}")
    return wanted


def load_questions(user_dir) -> list[Question]:
    user_dir = Path(user_dir)
    rows = json.loads((user_dir / "qa.json").read_text(encoding="utf-8-sig"))
    if isinstance(rows, dict):
        rows = rows.get("qa") or rows.get("questions") or rows.get("data")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No QA records in {user_dir / 'qa.json'}")
    result, seen = [], set()
    for row in rows:
        qid = str(row.get("query_id", row.get("id", "")))
        category = CATEGORY_ALIASES.get(row.get("category", row.get("c")))
        question = clean_text(row.get("question", row.get("q", "")))
        answer = row.get("answers", row.get("answer", row.get("a", "")))
        answers = tuple(clean_text(a) for a in (answer if isinstance(answer, list) else [answer]))
        if not qid or qid in seen or category is None or not question or not all(answers):
            raise ValueError(f"Invalid or duplicate QA record {qid!r} in {user_dir.name}")
        seen.add(qid)
        metadata = dict(row.get("metadata") or {})
        metadata.setdefault("unanswerable", bool(row.get("unanswerable", False)))
        result.append(Question(qid, user_dir.name, None, question, answers,
                               tuple(row.get("evidence_event_ids", ())), category, metadata))
    return result


def _cleaned_event(user, row, line_number):
    event = row.get("payload", {}).get("event", row.get("event", {})) or {}
    data = event.get("data") or {}
    cleaning = row.get("_cleaning") or {}
    text = data.get("vlm_caption") or cleaning.get("source_text")
    if not text:
        text = "\n".join(str(data[k]) for k in ("title", "content", "text", "description") if data.get(k))
    if not text:
        text = json.dumps(data, ensure_ascii=False, sort_keys=True)
    app = data.get("_effective_app") or event.get("from_app") or "未知应用"
    seconds = float(row.get("ts", cleaning.get("first_ts", 0)))
    if seconds > 10**11:
        seconds /= 1000
    timestamp = datetime.fromtimestamp(seconds, TZ).isoformat(timespec="milliseconds")
    eid = str(cleaning.get("observation_id") or f"observation:{line_number}")
    return MemoryEvent(eid, user, "observations", "user", f"[{timestamp}] [{app}] {text}", timestamp,
                       {"source_lines": cleaning.get("source_lines", []), "app": app,
                        "perception_type": event.get("perception_type"), "input_line": line_number})


def _raw_event(user, row, line_number):
    # observations.jsonl rows already carry the complete original record in content.
    return MemoryEvent(str(row["event_id"]), user, row.get("session_id", "observations"),
                       row.get("role", "user"), row["content"], row["timestamp"],
                       {**(row.get("metadata") or {}), "input_line": line_number})


def load_events(user_dir, input_kind: str = "cleaned", limit: int | None = None) -> list[MemoryEvent]:
    if input_kind not in INPUTS:
        raise ValueError(f"input must be one of {sorted(INPUTS)}")
    user_dir = Path(user_dir)
    source = user_dir / INPUTS[input_kind]
    convert = _cleaned_event if input_kind == "cleaned" else _raw_event
    events, seen = [], set()
    with source.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if limit is not None and len(events) >= limit:
                break
            event = convert(user_dir.name, json.loads(line), line_number)
            if event.event_id in seen:
                raise ValueError(f"Duplicate observation ID {event.event_id} in {source}")
            seen.add(event.event_id)
            events.append(event)
    return sorted(events, key=lambda e: e.timestamp)


def chinese_f1(prediction, references) -> float:
    """Han-character + Latin word/number unigram F1, max over references."""
    def tokenize(text):
        return re.findall(r"[㐀-鿿]|[a-z0-9]+(?:\.[0-9]+)?", clean_text(text).lower())
    p = tokenize(prediction)
    best = 0.0
    for ref in references:
        r = tokenize(ref)
        if not p or not r:
            best = max(best, float(p == r))
            continue
        common = sum((Counter(p) & Counter(r)).values())
        best = max(best, 2 * common / (len(p) + len(r)))
    return best
