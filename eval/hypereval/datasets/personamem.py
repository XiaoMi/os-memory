# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""PersonaMem dataset loader.

PersonaMem (Jiang et al., 2025, https://github.com/bowen-upenn/PersonaMem) is a
multiple-choice personalization benchmark: a user persona holds multi-session
chatbot conversations, and in-situ questions probe whether a model can recall
shared facts and track evolving preferences. Each question offers 4 candidate
assistant replies (``all_options``); exactly one (``correct_answer`` = ``(a)``..
``(d)``) is right. The official metric is multiple-choice accuracy, reported
overall and per question_type (7 fine-grained types).

Data files (put under ``data/personamem/``):
    questions_<size>.csv          — one row per question
    shared_contexts_<size>.jsonl  — one line per context: {context_id: [msg,...]}
where ``<size>`` is 32k / 128k / 1M. A message is an OpenAI-style dict with
``role`` (system/user/assistant) and ``content``.

Key structural fact this loader honours: within one shared context, different
questions read different PREFIXES of the message list — the ``question`` at row
``i`` sees ``context[:end_index_in_shared_context]``. To faithfully reproduce
the benchmark (a time-ordered preference question must not see future turns), we
emit ONE ``BenchmarkSample`` per distinct ``(shared_context_id, end_index)``
pair, whose events are exactly that prefix slice. Overlapping prefixes are
re-ingested per slice; that redundancy is the cost of faithfulness (the user
chose end_index slicing over whole-context ingest).

The runner scores multiple-choice questions via the MCQ branch (a question with
``metadata["all_options"]`` is treated as MCQ); this loader carries the options
and the correct option letter in ``Question.metadata`` and writes NO
answering/scoring logic itself (that lives in ``judge.py`` / ``runner.py``).
"""
from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path
from typing import Any

from hypereval.protocol import BenchmarkSample, MemoryEvent, Question

# ``all_options`` cells are long JSON/py-literal strings; the default csv field
# limit (128 KiB) is comfortably enough for 32k but we raise it for headroom.
csv.field_size_limit(10 ** 7)

# A synthetic monotonically increasing timestamp per message index, so the
# runner ingests events in conversation order (PersonaMem carries no per-turn
# timestamps). Only ordering matters, not the absolute values.
_BASE_EPOCH = 1_700_000_000  # 2023-11-14, arbitrary fixed anchor

_QUESTION_TYPE_ALIASES = {
    "recalling_facts_mentioned_by_the_user": "acknowledge_latest_user_preferences",
    "recalling_the_reasons_behind_previous_updates": "revisit_reasons_behind_preference_updates",
    "generalizing_to_new_scenarios": "generalize_to_new_scenarios",
    "track_full_preference_updates": "track_full_preference_evolution",
}


def _iso(index: int) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(_BASE_EPOCH + index, tz=timezone.utc).isoformat()


def _option_letter(correct_answer: str) -> str | None:
    """Extract the single option letter a-d from a ``correct_answer`` cell.

    Accepts ``(c)`` / ``c`` / ``C`` etc. Returns lowercase letter or None.
    """
    if not correct_answer:
        return None
    m = re.search(r"\(?\s*([a-dA-D])\s*\)?", correct_answer.strip())
    return m.group(1).lower() if m else None


def _canonical_question_type(value: str) -> str:
    question_type = value.strip()
    return _QUESTION_TYPE_ALIASES.get(question_type, question_type)


def _parse_options(raw: str) -> list[str]:
    """Parse the ``all_options`` cell into a list of option texts.

    The cell is a Python/JSON list literal of 4 strings, each usually prefixed
    with ``(a) ``..``(d) ``. We keep the full text (prefix included) so the
    answerer sees the same option labels the benchmark uses.
    """
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return []


def _option_text_for_letter(options: list[str], letter: str) -> str:
    """Return the plain text of the option whose label is ``(letter)``.

    Strips a leading ``(x) `` prefix if present. Falls back to positional index
    (a=0..d=3) when no option carries an explicit label.
    """
    tag = f"({letter})"
    for opt in options:
        stripped = opt.strip()
        if stripped.lower().startswith(tag):
            return stripped[len(tag):].strip()
    idx = "abcd".find(letter)
    if 0 <= idx < len(options):
        opt = options[idx].strip()
        return re.sub(r"^\([a-dA-D]\)\s*", "", opt).strip()
    return ""


def _contexts_path_for(questions_path: Path) -> Path:
    """Derive shared_contexts_<size>.jsonl from questions_<size>.csv."""
    name = questions_path.name
    if name.startswith("questions_") and name.endswith(".csv"):
        size = name[len("questions_"):-len(".csv")]
        return questions_path.with_name(f"shared_contexts_{size}.jsonl")
    # Fallback: same stem swap.
    return questions_path.with_name(
        questions_path.stem.replace("questions", "shared_contexts") + ".jsonl"
    )


def _load_contexts(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load {context_id: [message, ...]} from the jsonl file (one dict per line)."""
    contexts: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for context_id, messages in obj.items():
                contexts[context_id] = messages
    return contexts


class PersonaMemDataset:
    """PersonaMem multiple-choice personalization benchmark.

    ``run_tag="personamem"`` -> run dir named ``<system>_personamem_<sub>_<stamp>``.
    Data path (questions_<size>.csv) and sample size are declared in the system
    config's ``_data`` / ``_sample_size`` and read by run.py — not hardcoded here.
    """

    name = "personamem"
    run_tag = "personamem"

    def load(
        self, path: str | None = None, sample_size: int | None = None
    ) -> list[BenchmarkSample]:
        if not path:
            raise RuntimeError(
                "PersonaMemDataset.load requires data_path to point at "
                "questions_<size>.csv"
            )
        questions_path = Path(path).expanduser().resolve()
        if not questions_path.exists():
            raise FileNotFoundError(f"PersonaMem questions csv missing: {questions_path}")
        contexts_path = _contexts_path_for(questions_path)
        if not contexts_path.exists():
            raise FileNotFoundError(
                f"PersonaMem shared_contexts jsonl missing: {contexts_path}"
            )

        contexts = _load_contexts(contexts_path)

        # Group question rows by (shared_context_id, end_index). Each group
        # becomes one BenchmarkSample whose events are context[:end_index].
        with questions_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        groups: dict[tuple[str, int], list[dict[str, str]]] = {}
        for row in rows:
            context_id = row.get("shared_context_id") or ""
            try:
                end_index = int(row.get("end_index_in_shared_context") or 0)
            except ValueError:
                end_index = 0
            groups.setdefault((context_id, end_index), []).append(row)

        # Sort by end_index ascending so a small sample_size picks the shortest
        # prefixes first (cheapest to ingest for smoke runs).
        ordered_keys = sorted(groups.keys(), key=lambda k: (k[1], k[0]))

        samples: list[BenchmarkSample] = []
        context_events: dict[str, tuple[MemoryEvent, ...]] = {}
        for context_id, end_index in ordered_keys:
            messages = contexts.get(context_id)
            if not messages:
                # Question references a context we don't have; skip the group.
                continue
            sample_id = f"{context_id[:12]}__{end_index}"
            user_id = context_id[:12] or "persona"
            if context_id not in context_events:
                context_events[context_id] = tuple(
                    MemoryEvent(
                        event_id=f"{context_id}_e{idx}",
                        user_id=user_id,
                        session_id=context_id,
                        role=str(msg.get("role") or "user"),
                        content=str(msg.get("content") or ""),
                        timestamp=_iso(idx),
                        metadata={
                            **msg,
                            "event_time_ms": (_BASE_EPOCH + idx) * 1000,
                            "event_type": str(msg.get("role") or "user"),
                            "entity_ids": [],
                        },
                    )
                    for idx, msg in enumerate(messages)
                )
            full_events = context_events[context_id]
            prefix_length = end_index if end_index > 0 else len(full_events)
            events = full_events[:prefix_length]

            questions: list[Question] = []
            for qi, row in enumerate(groups[(context_id, end_index)]):
                options = _parse_options(row.get("all_options") or "")
                letter = _option_letter(row.get("correct_answer") or "")
                question_type = _canonical_question_type(row.get("question_type") or "")
                answer_text = (
                    _option_text_for_letter(options, letter) if letter else ""
                )
                questions.append(
                    Question(
                        query_id=str(row.get("question_id") or f"{sample_id}_q{qi}"),
                        user_id=user_id,
                        session_id=context_id,
                        question=str(row.get("user_question_or_message") or ""),
                        answers=(answer_text,) if answer_text else (),
                        evidence_event_ids=(),  # no native evidence labels
                        category=question_type or None,
                        metadata={
                            "all_options": options,
                            "correct_option": letter or "",
                            "correct_answer_raw": row.get("correct_answer") or "",
                            "topic": row.get("topic") or "",
                            "end_index": end_index,
                            "shared_context_id": context_id,
                        },
                    )
                )

            samples.append(
                BenchmarkSample(
                    sample_id=sample_id,
                    dataset="personamem",
                    events=events,
                    questions=tuple(questions),
                    metadata={
                        "shared_context_id": context_id,
                        "end_index": end_index,
                        "event_count": len(events),
                    },
                )
            )

        if sample_size is not None:
            samples = samples[:sample_size]
        return samples
