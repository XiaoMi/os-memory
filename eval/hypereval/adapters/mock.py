# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from hypereval.adapters.base import MemoryAdapter
from hypereval.protocol import MemoryEvent, MemoryRecord


class MockMemoryAdapter(MemoryAdapter):
    """Deterministic lexical retriever used to validate the whole platform."""

    def __init__(self) -> None:
        self._records: dict[str, list[MemoryEvent]] = {}

    def initialize(self, context: dict[str, Any]) -> None:
        self._records = {}

    def reset(self, user_id: str) -> None:
        self._records[user_id] = []

    def ingest(self, event: MemoryEvent) -> None:
        self._records.setdefault(event.user_id, []).append(event)

    def retrieve(
        self,
        query: str,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 10,
    ) -> list[MemoryRecord]:
        query_terms = self._terms(query)
        candidates = []
        for event in self._records.get(user_id, []):
            if session_id and event.session_id != session_id:
                continue
            content_terms = self._terms(event.content)
            overlap = len(query_terms & content_terms)
            score = overlap / max(len(query_terms), 1)
            if overlap:
                candidates.append((score, event))
        candidates.sort(key=lambda item: (item[0], item[1].timestamp), reverse=True)
        return [
            MemoryRecord(
                memory_id=event.event_id,
                content=event.content,
                score=score,
                source_event_ids=(event.event_id,),
                metadata={"role": event.role},
            )
            for score, event in candidates[:top_k]
        ]

    def dump_state(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "mock_state.json").write_text(
            str({user: [event.event_id for event in events] for user, events in self._records.items()}),
            encoding="utf-8",
        )

    @staticmethod
    def _terms(text: str) -> set[str]:
        return set(re.findall(r"[\w]+", text.lower()))
