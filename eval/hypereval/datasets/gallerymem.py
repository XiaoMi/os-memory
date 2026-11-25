# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""Mem-Gallery dataset loader.

Mem-Gallery is a multimodal, multi-session conversational-memory benchmark.
The released files contain one scenario per JSON file.  Each scenario has
historical dialogue rounds (some with images) and a ``human-annotated QAs``
list with nine task types.  We keep image ids, paths, and captions in both the
event content and metadata so text adapters can use captions without silently
discarding the visual evidence.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hypereval.protocol import BenchmarkSample, MemoryEvent, Question


def _to_epoch_ms(value: Any) -> int:
    text = str(value or "").strip()
    if not text:
        return 0
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp() * 1000)
    except ValueError:
        return 0


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _resolve_image(root: Path, raw_path: str) -> str:
    path = Path(raw_path)
    if path.is_absolute():
        return str(path)
    # Official paths are ../image/<scenario>/<filename>, relative to data/dialog.
    if raw_path.startswith("../image/"):
        return str((root / "image" / raw_path[len("../image/"):]).resolve())
    return str((root / "image" / raw_path).resolve())


class GalleryMemDataset:
    """Load all scenarios from a Mem-Gallery ``data`` directory."""

    name = "gallerymem"
    run_tag = "gallerymem"

    def load(
        self, path: str | Path | None = None, sample_size: int | None = None
    ) -> list[BenchmarkSample]:
        candidate = Path(path).expanduser().resolve() if path is not None else None
        if candidate is not None and candidate.is_file():
            data_root = candidate.parent.parent
            return [self._load_scenario(candidate, data_root)]
        data_root = self._resolve_root(path)
        dialog_dir = data_root / "dialog"
        if not dialog_dir.exists():
            raise FileNotFoundError(f"Mem-Gallery dialog directory missing: {dialog_dir}")
        files = sorted(dialog_dir.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No Mem-Gallery scenario JSON files in {dialog_dir}")

        samples = [self._load_scenario(file_path, data_root) for file_path in files]
        if sample_size is not None:
            return samples[:sample_size]
        return samples

    @staticmethod
    def _resolve_root(path: str | Path | None) -> Path:
        if path is None:
            raise ValueError(
                "Mem-Gallery requires --data-path pointing to its data directory"
            )
        root = Path(path).expanduser().resolve()
        if root.name.lower() == "data" and (root / "dialog").exists():
            return root
        if (root / "data" / "dialog").exists():
            return root / "data"
        return root

    def _load_scenario(self, path: Path, data_root: Path) -> BenchmarkSample:
        payload = json.loads(path.read_text(encoding="utf-8"))
        profile = payload.get("character_profile") or {}
        scenario = path.stem
        character_name = str(profile.get("name") or scenario)
        sample_id = scenario
        user_id = scenario
        events: list[MemoryEvent] = []
        round_to_event: dict[str, str] = {}

        for session in payload.get("multi_session_dialogues") or []:
            session_id = str(session.get("session_id") or "session")
            date = str(session.get("date") or "")
            event_time_ms = _to_epoch_ms(date)
            for index, dialogue in enumerate(session.get("dialogues") or [], start=1):
                if not isinstance(dialogue, dict):
                    continue
                round_id = str(dialogue.get("round") or f"{session_id}:{index}")
                user_text = str(dialogue.get("user") or "").strip()
                assistant_text = str(dialogue.get("assistant") or "").strip()
                image_ids = _as_list(dialogue.get("image_id"))
                image_paths = [
                    _resolve_image(data_root, value)
                    for value in _as_list(dialogue.get("input_image"))
                ]
                captions = _as_list(dialogue.get("image_caption"))
                parts = []
                if user_text:
                    parts.append(f"user ({character_name}): {user_text}")
                if assistant_text:
                    parts.append(f"assistant: {assistant_text}")
                if image_ids:
                    parts.append("image_id: " + ", ".join(image_ids))
                if captions:
                    parts.append("image_caption: " + " | ".join(captions))
                content = "\n".join(parts)
                event_id = f"{sample_id}-{round_id.replace(':', '-') }"
                round_to_event[round_id] = event_id
                events.append(
                    MemoryEvent(
                        event_id=event_id,
                        user_id=user_id,
                        session_id=session_id,
                        role="user_assistant",
                        content=content,
                        timestamp=date,
                        metadata={
                            "scenario": scenario,
                            "round": round_id,
                            "date": date,
                            "event_time_ms": event_time_ms,
                            "image_ids": image_ids,
                            "image_paths": image_paths,
                            "image_captions": captions,
                        },
                    )
                )

        questions: list[Question] = []
        for index, item in enumerate(payload.get("human-annotated QAs") or []):
            if not isinstance(item, dict) or not item.get("question"):
                continue
            clues = _as_list(item.get("clue"))
            evidence = tuple(round_to_event.get(clue, clue) for clue in clues)
            question_images = _as_list(item.get("question_image"))
            question_image_paths = [
                _resolve_image(data_root, value) for value in question_images
            ]
            question_caption = item.get("image_caption")
            session_ids = _as_list(item.get("session_id"))
            answer = item.get("answer")
            answers = answer if isinstance(answer, list) else [answer]
            metadata = {
                "dataset": self.name,
                "point": str(item.get("point") or "").upper(),
                "question_image_paths": question_image_paths,
                "question_image_caption": question_caption or "",
                "question_session_ids": session_ids,
                "clue_rounds": clues,
            }
            questions.append(
                Question(
                    query_id=f"{sample_id}_q{index}",
                    user_id=user_id,
                    session_id=None,
                    question=str(item["question"]),
                    answers=tuple(str(value) for value in answers if value is not None),
                    evidence_event_ids=evidence,
                    category=str(item.get("point") or "").upper() or None,
                    metadata=metadata,
                )
            )

        return BenchmarkSample(
            sample_id=sample_id,
            dataset=self.name,
            events=tuple(events),
            questions=tuple(questions),
            metadata={
                "scenario": scenario,
                "character_name": character_name,
                "character_profile": profile,
                "event_count": len(events),
                "question_count": len(questions),
            },
        )
