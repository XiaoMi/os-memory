# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""LoCoMo dataset loader.

The released LoCoMo JSON stores one conversation as session_1, session_2 ...
arrays. Each conversation becomes one isolated benchmark episode and evidence
references such as D1:3 are mapped to event ids.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hypereval.protocol import BenchmarkSample, MemoryEvent, Question


_SESSION_KEY = re.compile(r"^session_(\d+)$")


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
    for fmt in (
        "%I:%M %p on %d %B, %Y",
        "%d %B %Y",
        "%B %d, %Y",
        "%Y-%m-%d",
    ):
        try:
            parsed = datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            return int(parsed.timestamp() * 1000)
        except ValueError:
            pass
    return 0


def _session_items(conv: dict[str, Any]) -> list[tuple[str, str, list[dict[str, Any]]]]:
    conversation = conv.get("conversation")
    if isinstance(conversation, list):
        return [("session_1", str(conv.get("date_time") or ""), conversation)]
    turns = conv.get("turns")
    if isinstance(turns, list):
        return [("session_1", str(conv.get("date_time") or ""), turns)]
    source = conversation if isinstance(conversation, dict) else conv
    items: list[tuple[str, str, list[dict[str, Any]]]] = []
    for key, value in source.items():
        match = _SESSION_KEY.match(str(key))
        if not match or not isinstance(value, list):
            continue
        date = source.get(f"{key}_date_time") or source.get(f"{key}_date") or ""
        items.append((str(key), str(date), value))
    return sorted(items, key=lambda item: int(item[0].split("_")[-1]))


def _emit_events(
    sample_id: str, user_id: str, conv: dict[str, Any]
) -> tuple[list[MemoryEvent], dict[str, str]]:
    events: list[MemoryEvent] = []
    evidence_map: dict[str, str] = {}
    for session_id, session_date, turns in _session_items(conv):
        session_ms = _to_epoch_ms(session_date)
        for index, turn in enumerate(turns, start=1):
            if not isinstance(turn, dict):
                continue
            text = turn.get("text") or turn.get("content") or ""
            speaker = turn.get("speaker") or turn.get("role") or (
                "user" if index % 2 else "assistant"
            )
            event_id = f"{sample_id}-{session_id}-e{index}"
            dia_id = str(turn.get("dia_id") or "")
            content = str(text)
            event = MemoryEvent(
                event_id=event_id,
                user_id=user_id,
                session_id=str(turn.get("session_id") or session_id),
                role=str(speaker),
                content=content,
                timestamp=str(turn.get("timestamp") or session_date),
                metadata={
                    "dia_id": dia_id,
                    "session_date": session_date,
                    "event_time_ms": _to_epoch_ms(turn.get("timestamp") or session_date)
                    or session_ms,
                },
            )
            events.append(event)
            if dia_id:
                evidence_map[dia_id] = event_id
            if content:
                evidence_map.setdefault(content, event_id)
    return events, evidence_map


def _emit_questions(
    sample_id: str,
    user_id: str,
    conv: dict[str, Any],
    evidence_map: dict[str, str],
) -> list[Question]:
    questions: list[Question] = []
    for index, item in enumerate(conv.get("qa") or conv.get("questions") or []):
        if not isinstance(item, dict):
            continue
        answer = item.get("answer")
        answers = item.get("answers") or ([answer] if answer is not None else [])
        raw_evidence = item.get("evidence") or item.get("evidence_event_ids") or ()
        evidence = tuple(evidence_map.get(str(value), str(value)) for value in raw_evidence)
        category = item.get("category")
        questions.append(
            Question(
                query_id=f"{sample_id}_q{index}",
                user_id=user_id,
                session_id=str(item["session_id"]) if item.get("session_id") else None,
                question=str(item.get("question") or ""),
                answers=tuple(str(value) for value in answers),
                evidence_event_ids=evidence,
                category=str(category) if category is not None else None,
                metadata={
                    "dataset": "locomo",
                    "adversarial_answer": str(item.get("adversarial_answer") or ""),
                    "unanswerable": str(category) == "5",
                },
            )
        )
    return questions


class LoCoMoDataset:
    """Load one isolated conversation per BenchmarkSample episode."""

    name = "locomo"
    run_tag = "locomo"

    def load(self, path: str | None = None, sample_size: int | None = None) -> list[BenchmarkSample]:
        if not path:
            raise RuntimeError("LoCoMoDataset.load requires --data-path")
        data_path = Path(path)
        if not data_path.exists():
            raise FileNotFoundError(f"LoCoMo data missing: {data_path}")
        payload = json.loads(data_path.read_text(encoding="utf-8"))
        conversations = payload if isinstance(payload, list) else [payload]
        samples: list[BenchmarkSample] = []
        for index, conv in enumerate(conversations):
            if not isinstance(conv, dict):
                continue
            sample_id = str(conv.get("sample_id") or conv.get("id") or f"locomo_{index}")
            conversation = conv.get("conversation")
            conversation_speaker = conversation.get("speaker_a") if isinstance(conversation, dict) else None
            user_id = str(conv.get("user_id") or conv.get("speaker_a") or conversation_speaker or sample_id)
            events, evidence_map = _emit_events(sample_id, user_id, conv)
            questions = _emit_questions(sample_id, user_id, conv, evidence_map)
            samples.append(
                BenchmarkSample(
                    sample_id=sample_id,
                    dataset=self.name,
                    events=tuple(events),
                    questions=tuple(questions),
                    metadata={"episode_id": sample_id, "user_id": user_id},
                )
            )
        return samples[:sample_size] if sample_size is not None else samples
