# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""LongMem-Eval dataset loader.

LongMem-Eval defines one evaluation instance as one complete episode: all
haystack sessions are ingested before the question is asked. A separate
BenchmarkSample is emitted for every question so each adapter can isolate its
state per question and never leak memories across episodes.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hypereval.protocol import BenchmarkSample, MemoryEvent, Question


def _to_epoch_ms(value: Any) -> int:
    if value is None or value == "":
        return 0
    if isinstance(value, (int, float)):
        return int(value if value > 10_000_000_000 else value * 1000)
    text = str(value).strip()
    for candidate in (text, text.replace("Z", "+00:00")):
        try:
            parsed = datetime.fromisoformat(candidate)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return int(parsed.timestamp() * 1000)
        except ValueError:
            pass
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%B %d, %Y"):
        try:
            parsed = datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            return int(parsed.timestamp() * 1000)
        except ValueError:
            pass
    return 0


def _read_payload(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, list) else [payload]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


class LongMemEvalDataset:
    name = "longmemeval"
    run_tag = "longmemeval"

    def load(self, path: str | None = None, sample_size: int | None = None) -> list[BenchmarkSample]:
        if not path:
            raise RuntimeError("LongMemEvalDataset.load requires --data-path")
        data_path = Path(path)
        if not data_path.exists():
            raise FileNotFoundError(f"LongMem-Eval data missing: {data_path}")

        samples: list[BenchmarkSample] = []
        for index, item in enumerate(_read_payload(data_path)):
            question_id = str(item.get("question_id") or item.get("id") or f"longmemeval_{index}")
            user_id = str(item.get("user_id") or question_id)
            session_ids = [str(value) for value in item.get("haystack_session_ids") or []]
            session_dates = [str(value) for value in item.get("haystack_dates") or []]
            sessions = item.get("haystack_sessions") or []
            answer_sessions = {str(value) for value in item.get("answer_session_ids") or []}
            events: list[MemoryEvent] = []
            evidence_ids: list[str] = []

            for session_index, session in enumerate(sessions):
                session_id = (
                    session_ids[session_index]
                    if session_index < len(session_ids)
                    else f"session_{session_index + 1}"
                )
                session_date = (
                    session_dates[session_index]
                    if session_index < len(session_dates)
                    else ""
                )
                session_ms = _to_epoch_ms(session_date)
                if not isinstance(session, list):
                    continue
                for turn_index, turn in enumerate(session, start=1):
                    if not isinstance(turn, dict):
                        continue
                    content = str(turn.get("content") or turn.get("text") or "")
                    event_id = f"{question_id}-{session_id}-e{turn_index}"
                    event = MemoryEvent(
                        event_id=event_id,
                        user_id=user_id,
                        session_id=session_id,
                        role=str(turn.get("role") or "user"),
                        content=content,
                        timestamp=session_date,
                        metadata={
                            "question_id": question_id,
                            "session_date": session_date,
                            "event_time_ms": session_ms,
                            "has_answer": bool(turn.get("has_answer")),
                            "answer_session": session_id in answer_sessions,
                        },
                    )
                    events.append(event)
                    if turn.get("has_answer") and event.role == "user":
                        evidence_ids.append(event_id)

            answer = item.get("answer")
            answers = answer if isinstance(answer, list) else [answer]
            question_type = str(item.get("question_type") or "") or None
            abstention = question_id.endswith("_abs")
            samples.append(
                BenchmarkSample(
                    sample_id=question_id,
                    dataset=self.name,
                    events=tuple(events),
                    questions=(
                        Question(
                            query_id=question_id,
                            user_id=user_id,
                            session_id=None,
                            question=str(item.get("question") or ""),
                            answers=tuple(str(value) for value in answers if value is not None),
                            evidence_event_ids=tuple(evidence_ids),
                            category=question_type,
                            metadata={
                                "dataset": self.name,
                                "question_type": question_type,
                                "question_date": item.get("question_date"),
                                "abstention": abstention,
                                "answer_session_ids": sorted(answer_sessions),
                                "has_user_evidence": bool(evidence_ids),
                            },
                        ),
                    ),
                    metadata={
                        "episode_id": question_id,
                        "user_id": user_id,
                        "question_type": question_type,
                        "question_date": item.get("question_date"),
                    },
                )
            )
        return samples[:sample_size] if sample_size is not None else samples
