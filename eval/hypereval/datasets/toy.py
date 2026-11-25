# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""Minimal ToyDataset placeholder. Not used by os_events path.

Kept so hypereval.registry can still import it (LoCoMo/ToY were referenced in
the original registry). A real ToyDataset was not shipped in this snapshot;
we only need the class to exist for import symmetry.
"""
from __future__ import annotations

from typing import Any

from hypereval.protocol import BenchmarkSample, MemoryEvent, Question


class ToyDataset:
    """Two-event toy dataset for smoke testing hypereval plumbing."""

    def load(self, path: str | None = None, sample_size: int | None = None) -> list[BenchmarkSample]:
        events = (
            MemoryEvent(
                event_id="toy_e1",
                user_id="u1",
                session_id="s1",
                role="user",
                content="I moved to Shanghai last week.",
                timestamp="2026-07-30T10:00:00",
            ),
            MemoryEvent(
                event_id="toy_e2",
                user_id="u1",
                session_id="s1",
                role="assistant",
                content="Got it, welcome to Shanghai.",
                timestamp="2026-07-30T10:00:05",
            ),
        )
        questions = (
            Question(
                query_id="toy_q1",
                user_id="u1",
                session_id="s1",
                question="Where does the user live now?",
                answers=("Shanghai",),
                evidence_event_ids=("toy_e1",),
            ),
        )
        return [BenchmarkSample(sample_id="toy1", dataset="toy", events=events, questions=questions)]
