# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""MemBench trajectory loader.

MemBench stores one multiple-choice question with the conversation trajectory
that precedes it.  The loader converts each trajectory into an isolated
benchmark sample so every memory system is reset between trajectories.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from hypereval.protocol import BenchmarkSample, MemoryEvent, Question


class MemBenchDataset:
    """Load the four primary MemBench data files.

    ``path`` may point at ``MemData`` or its parent checkout directory.  When
    sampling, samples are selected evenly across the four primary files rather
    than taking all samples from the first file.
    """

    run_tag = "membench"
    _GROUPS = (
        ("FirstAgent", "participation_reflective", ("highlevel.json", "highlevel_rec.json")),
        (
            "FirstAgent",
            "participation_factual",
            (
                "aggregative.json",
                "comparative.json",
                "conditional.json",
                "knowledge_update.json",
                "post_processing.json",
                "simple.json",
                "lowlevel_rec.json",
                "RecMultiSession.json",
            ),
        ),
        ("ThirdAgent", "observation_reflective", ("highlevel.json",)),
        (
            "ThirdAgent",
            "observation_factual",
            (
                "aggregative.json",
                "comparative.json",
                "conditional.json",
                "knowledge_update.json",
                "post_processing.json",
                "simple.json",
            ),
        ),
    )

    def load(self, path: str | Path | None = None, sample_size: int | None = None) -> list[BenchmarkSample]:
        root = self._resolve_root(path)
        grouped: list[list[BenchmarkSample]] = []
        for agent_dir, group_name, filenames in self._GROUPS:
            group_samples: list[BenchmarkSample] = []
            for filename in filenames:
                file_path = root / agent_dir / filename
                if not file_path.exists():
                    raise FileNotFoundError(f"MemBench data file does not exist: {file_path}")
                group_samples.extend(self._load_file(file_path, agent_dir, group_name))
            grouped.append(group_samples)

        if sample_size is None:
            return [sample for group in grouped for sample in group]
        if sample_size <= 0:
            return []

        base, remainder = divmod(sample_size, len(grouped))
        selected: list[BenchmarkSample] = []
        for index, group in enumerate(grouped):
            count = base + (1 if index < remainder else 0)
            selected.extend(group[:count])
        return selected

    @staticmethod
    def _resolve_root(path: str | Path | None) -> Path:
        if path is None:
            raise ValueError("MemBench requires --data-path pointing to the MemData directory")
        candidate = Path(path).expanduser().resolve()
        if candidate.name.lower() == "memdata":
            return candidate
        nested = candidate / "MemData"
        if nested.exists():
            return nested
        return candidate

    def _load_file(self, path: Path, agent_dir: str, group_name: str) -> list[BenchmarkSample]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        samples: list[BenchmarkSample] = []
        file_stem = path.stem
        for question_type, trajectories_by_role in payload.items():
            if isinstance(trajectories_by_role, list):
                role_trajectories = [(str(question_type), trajectories_by_role)]
            elif isinstance(trajectories_by_role, dict):
                role_trajectories = list(trajectories_by_role.items())
            else:
                continue
            for role_name, trajectories in role_trajectories:
                if not isinstance(trajectories, list):
                    continue
                for trajectory in trajectories:
                    sample = self._convert_trajectory(
                        trajectory,
                        agent_dir=agent_dir,
                        file_stem=file_stem,
                        group_name=group_name,
                        question_type=str(question_type),
                        role_name=str(role_name),
                    )
                    if sample is not None:
                        samples.append(sample)
        return samples

    def _convert_trajectory(
        self,
        trajectory: Any,
        *,
        agent_dir: str,
        file_stem: str,
        group_name: str,
        question_type: str,
        role_name: str,
    ) -> BenchmarkSample | None:
        if not isinstance(trajectory, dict) or not isinstance(trajectory.get("QA"), dict):
            return None
        qa = trajectory["QA"]
        trajectory_id = str(trajectory.get("tid", len(str(trajectory))))
        sample_id = f"{agent_dir}-{file_stem}-{question_type}-{role_name}-{trajectory_id}"
        message_list = trajectory.get("message_list")
        events = tuple(self._events(message_list, sample_id, agent_dir))
        evidence_event_ids = self._target_event_ids(
            qa.get("target_step_id"), message_list, sample_id
        )
        choices = qa.get("choices") or {}
        options = []
        for letter in ("A", "B", "C", "D"):
            if letter not in choices:
                continue
            value = choices[letter]
            if isinstance(value, list):
                value = "; ".join(str(item) for item in value)
            options.append(f"({letter.lower()}) {value}")
        question_text = str(qa.get("question") or "")
        question_time = self._normalize_timestamp(qa.get("time"))
        if question_time:
            question_text = f"{question_text} (current time is {question_time})"
        correct_option = str(qa.get("ground_truth") or "").strip().lower()
        raw_answer = qa.get("answer")
        if isinstance(raw_answer, list):
            answers = tuple(str(item) for item in raw_answer)
        elif raw_answer:
            answers = (str(raw_answer),)
        else:
            answers = ()
        question = Question(
            query_id=f"{sample_id}-q0",
            user_id=sample_id,
            session_id=sample_id,
            question=question_text,
            answers=answers,
            evidence_event_ids=evidence_event_ids,
            category=f"{group_name}/{question_type}",
            metadata={
                "all_options": options,
                "correct_option": correct_option,
                "question_time": question_time,
                "target_step_id": qa.get("target_step_id"),
                "source_file": f"{agent_dir}/{file_stem}.json",
                "agent": agent_dir,
            },
        )
        return BenchmarkSample(
            sample_id=sample_id,
            dataset="membench",
            events=events,
            questions=(question,),
            metadata={
                "agent": agent_dir,
                "file": file_stem,
                "group": group_name,
                "question_type": question_type,
                "role": role_name,
                "trajectory_id": trajectory_id,
            },
        )

    @staticmethod
    def _normalize_timestamp(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        match = re.search(r"(\d{4}[-/]\d{2}[-/]\d{2}\s+\d{2}:\d{2})", text)
        if match:
            text = match.group(1)
            for fmt in ("%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M"):
                try:
                    return datetime.strptime(text, fmt).isoformat(timespec="minutes")
                except ValueError:
                    pass
        return text

    @staticmethod
    def _events(message_list: Any, sample_id: str, agent_dir: str) -> Iterable[MemoryEvent]:
        if not isinstance(message_list, list):
            return ()
        flattened: list[Any] = []
        for item in message_list:
            if isinstance(item, list):
                flattened.extend(item)
            else:
                flattened.append(item)
        events: list[MemoryEvent] = []
        for index, item in enumerate(flattened):
            if isinstance(item, dict) and ("user_message" in item or "user" in item):
                user_message = item.get("user_message", item.get("user", ""))
                assistant_message = item.get("assistant_message", item.get("assistant", ""))
                content = (
                    f"{index}[|]'user': {user_message}; "
                    f"'agent': {assistant_message}"
                )
                timestamp = MemBenchDataset._normalize_timestamp(item.get("time"))
                role = "user_assistant"
            elif isinstance(item, dict):
                content = f"{index}[|]{item.get('message', '')}"
                timestamp = MemBenchDataset._normalize_timestamp(item.get("time"))
                role = "user"
            else:
                content = f"{index}[|]{item}"
                timestamp = ""
                role = "user"
            events.append(
                MemoryEvent(
                    event_id=f"{sample_id}-e{index}",
                    user_id=sample_id,
                    session_id=sample_id,
                    role=role,
                    content=content,
                    timestamp=timestamp,
                    metadata={"agent": agent_dir, "step": index},
                )
            )
        return events

    @staticmethod
    def _target_event_ids(
        target_step_ids: Any, message_list: Any, sample_id: str
    ) -> tuple[str, ...]:
        if not isinstance(message_list, list) or not isinstance(target_step_ids, list):
            return ()
        coordinate_to_flat: dict[tuple[int, int], int] = {}
        flat_index = 0
        for outer_index, item in enumerate(message_list):
            if isinstance(item, list):
                for inner_index, _ in enumerate(item):
                    coordinate_to_flat[(inner_index, outer_index)] = flat_index
                    flat_index += 1
            else:
                coordinate_to_flat[(outer_index, 0)] = flat_index
                flat_index += 1
        event_ids: list[str] = []
        for target in target_step_ids:
            flat_target: int | None = None
            if isinstance(target, int):
                flat_target = target
            elif isinstance(target, (list, tuple)) and len(target) >= 2:
                try:
                    flat_target = coordinate_to_flat.get((int(target[0]), int(target[1])))
                except (TypeError, ValueError):
                    flat_target = None
            if flat_target is not None and 0 <= flat_target < flat_index:
                event_id = f"{sample_id}-e{flat_target}"
                if event_id not in event_ids:
                    event_ids.append(event_id)
        return tuple(event_ids)
