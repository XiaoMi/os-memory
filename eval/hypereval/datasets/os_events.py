# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""OsEventsDataset: load a phone OS user-behavior event stream from a sqlite dump.

Table schema of the device event SQLite dump:
    event_id TEXT, domain_id TEXT, source_long_id TEXT, event_type TEXT,
    title TEXT, description TEXT, tags TEXT (JSON),
    start_time INTEGER, end_time INTEGER, event_time INTEGER (ms),
    file_path TEXT, raw_data TEXT (JSON), extra TEXT (JSON),
    clue_state/vector_state/graph_state INTEGER

We collapse the whole DB into ONE BenchmarkSample with no Question objects.
The runner will call adapter.ingest for every event, then adapter.finalize_ingest,
and skip the QA loop because ``questions`` is empty. That gives us exactly the
"process the event stream once and dump memory state" behaviour we want.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from hypereval.protocol import BenchmarkSample, MemoryEvent


# Beijing time — the device DB uses ms epoch; renders as ISO for readability.
_BEIJING = timezone(timedelta(hours=8))


def _iso(ms: int) -> str:
    try:
        return datetime.fromtimestamp(ms / 1000.0, tz=_BEIJING).isoformat()
    except (OSError, OverflowError, ValueError):
        return ""


def _safe_json(text: str | None) -> Any:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _open_db_readonly(db_path: Path) -> sqlite3.Connection:
    """Open the sqlite file read-only.

    If sibling ``-shm`` / ``-wal`` files exist we make a *copy* first and open
    the copy read-only. This avoids WAL mutation on the shared JuiceFS mount.
    """
    tmp = Path(tempfile.mkdtemp(prefix="os_events_"))
    dst = tmp / db_path.name
    shutil.copy2(db_path, dst)
    for suffix in ("-shm", "-wal"):
        side = db_path.with_name(db_path.name + suffix)
        if side.exists():
            shutil.copy2(side, dst.with_name(dst.name + suffix))
    uri = f"file:{dst}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


class OsEventsDataset:
    """Turn a phone-OS user-behavior event sqlite dump into one BenchmarkSample.

    ``run_tag="os"`` -> run dir named ``<system>_os_<sub>_<stamp>``.
    Data path and sample size are declared by the consuming system config
    (``_data`` / ``_sample_size``), read by run.py — not hardcoded here.
    """

    name = "os_events"
    run_tag = "os"

    #: Which event_type values we keep. Everything else is filtered out at
    #: load time. Set to None to keep all rows.
    KEEP_EVENT_TYPES: set[str] | None = None

    def load(
        self,
        path: str | None = None,
        sample_size: int | None = None,
    ) -> list[BenchmarkSample]:
        if not path:
            raise RuntimeError(
                "OsEventsDataset.load requires data_path to point at "
                "device_data_event.db"
            )
        db_path = Path(path).expanduser().resolve()
        if not db_path.exists():
            raise FileNotFoundError(f"os_events db missing: {db_path}")

        conn = _open_db_readonly(db_path)
        cur = conn.cursor()
        rows = cur.execute(
            """
            SELECT event_id, domain_id, source_long_id, event_type,
                   title, description, tags, start_time, end_time, event_time,
                   file_path, raw_data, extra
            FROM event
            WHERE event_time > 0
            ORDER BY event_time ASC
            """
        ).fetchall()
        conn.close()

        events: list[MemoryEvent] = []
        user_id = "os_user"
        session_id = "device_stream"
        for row in rows:
            event_type = row["event_type"] or ""
            if self.KEEP_EVENT_TYPES is not None and event_type not in self.KEEP_EVENT_TYPES:
                continue
            title = (row["title"] or "").strip()
            description = (row["description"] or "").strip()
            # role carries the event_type so adapters can group by kind if useful.
            content_parts = [p for p in (title, description) if p]
            content = "\n".join(content_parts) or (row["source_long_id"] or "")
            extra_json = _safe_json(row["extra"])
            entities = []
            events.append(
                MemoryEvent(
                    event_id=str(row["event_id"]),
                    user_id=user_id,
                    session_id=session_id,
                    role=event_type or "event",
                    content=content,
                    timestamp=_iso(row["event_time"]),
                    metadata={
                        "source_long_id": row["source_long_id"],
                        "domain_id": row["domain_id"],
                        "event_type": event_type,
                        "start_time_ms": row["start_time"],
                        "end_time_ms": row["end_time"],
                        "event_time_ms": row["event_time"],
                        "file_path": row["file_path"] or "",
                        "tags": _safe_json(row["tags"]),
                        "extra": extra_json,
                        "entities": entities,
                        "entity_ids": [e["id"] for e in entities],
                    },
                )
            )

        if sample_size is not None:
            events = _stratified_head(events, sample_size)

        return [
            BenchmarkSample(
                sample_id="os_events",
                dataset="os_events",
                events=tuple(events),
                questions=(),  # runner will skip the QA loop -> pure ingest+dump
                metadata={"source_db": str(db_path), "event_count": len(events)},
            )
        ]


def _stratified_head(events: list[MemoryEvent], k: int) -> list[MemoryEvent]:
    """Return the first ``k`` events with roughly equal quota per event_type.

    Events must already be sorted by event_time. Each type gets a rolling
    quota; a type only starts consuming its quota once k events have been
    surveyed. Preserves temporal order within the selection so relation
    extraction sees a coherent timeline.
    """
    if not events or k <= 0:
        return events[:k]
    from collections import Counter

    types = Counter(e.role or "event" for e in events)
    n_types = len(types)
    per_type_target = max(1, k // n_types)
    quota = {t: per_type_target for t in types}
    # give leftover slots to the largest types
    remaining = k - per_type_target * n_types
    for t, _ in types.most_common():
        if remaining <= 0:
            break
        quota[t] += 1
        remaining -= 1
    picked: list[MemoryEvent] = []
    used = {t: 0 for t in types}
    for e in events:
        t = e.role or "event"
        if used[t] < quota[t]:
            picked.append(e)
            used[t] += 1
        if len(picked) >= k:
            break
    return picked
