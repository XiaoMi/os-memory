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

import json
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from hypereval.judge import LLMEvaluator
from hypereval.model_config import api_key, base_url, model_name
from hypereval.openai_compat import install_openai_gateway_compat
from hypereval.protocol import MemoryEvent
from hypereval.registry import create_adapter


_CL100K_ENCODING = None


def run_membench_capacity(
    system_name: str,
    data_path: str | Path,
    output_dir: str | Path,
    top_k: int = 10,
    sample_size: int | None = None,
    system_context: dict[str, Any] | None = None,
    resume: bool = False,
    retry_rounds: int = 2,
) -> Path:
    install_openai_gateway_compat()
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    payload = json.loads(Path(data_path).read_text(encoding="utf-8"))
    samples = list(payload.get("samples") or payload)
    if sample_size is not None:
        samples = samples[:sample_size]

    config = _load_system_config(system_context)
    evaluator = LLMEvaluator(
        model=model_name(config, "azure_openai/gpt-4o-mini"),
        api_key=api_key(config, "OPENAI_API_KEY"),
        base_url=base_url(config),
    )
    adapter = create_adapter(system_name)
    adapter.initialize(
        {
            "run_dir": str(root),
            "system": system_name,
            "dataset": "membench",
            **(system_context or {}),
        }
    )

    predictions_path = root / "capacity_predictions.jsonl"
    status_path = root / "sample_status.json"
    statuses = _load_statuses(status_path) if resume else {}
    completed_ids = {
        sample_id
        for sample_id, status in statuses.items()
        if status.get("status") == "success"
    }
    results = _load_completed_results(predictions_path, completed_ids) if resume else []
    predictions_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in results),
        encoding="utf-8",
    )
    with predictions_path.open("a", encoding="utf-8") as output:
        try:
            for sample in samples:
                sample_id = str(sample.get("sample_id") or sample.get("tid"))
                if sample_id in completed_ids:
                    continue
                attempts = max(1, int(retry_rounds) + 1)
                for attempt in range(1, attempts + 1):
                    try:
                        sample_results = _run_capacity_sample(
                            adapter,
                            evaluator,
                            sample,
                            sample_id,
                            top_k,
                        )
                        for item in sample_results:
                            output.write(json.dumps(item, ensure_ascii=False) + "\n")
                        output.flush()
                        results.extend(sample_results)
                        statuses[sample_id] = {
                            "status": "success",
                            "checkpoints": len(sample_results),
                            "attempt": attempt,
                        }
                        completed_ids.add(sample_id)
                        break
                    except Exception as exc:
                        statuses[sample_id] = {
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                            "attempt": attempt,
                        }
                        if attempt == attempts:
                            break
                _write_json_atomic(status_path, statuses)
                _write_capacity_metrics(root, results, statuses, len(samples))
        finally:
            adapter.dump_state(root / "system_state")
            adapter.close()

    _write_capacity_metrics(root, results, statuses, len(samples))
    _write_json_atomic(status_path, statuses)
    return root


def _run_capacity_sample(
    adapter,
    evaluator: LLMEvaluator,
    sample: dict[str, Any],
    sample_id: str,
    top_k: int,
) -> list[dict[str, Any]]:
    user_id = sample_id
    trajectory = sample.get("trajectory") or sample
    messages = list(trajectory.get("message_list") or [])
    events = [
        _capacity_event(sample_id, index, message)
        for index, message in enumerate(messages)
    ]
    qa = dict(trajectory.get("QA") or {})
    target_ids = [
        int(value)
        for value in qa.get("target_step_id") or []
        if isinstance(value, (int, float))
    ]
    if not target_ids:
        raise ValueError("capacity sample has no flat target_step_id")
    last_evidence = max(target_ids)
    token_count = 0
    checkpoint_count = 0
    sample_results: list[dict[str, Any]] = []
    streaming = bool(getattr(adapter, "supports_capacity_streaming", False))
    if streaming:
        adapter.begin_sample(sample_id)
        adapter.reset(user_id)
    for index, (message, event) in enumerate(zip(messages, events)):
        token_count += _cl100k_tokens(message)
        write_seconds = 0.0
        if streaming:
            write_started = time.perf_counter()
            adapter.ingest_capacity_event(event)
            write_seconds = time.perf_counter() - write_started
        if index < last_evidence:
            continue

        if not streaming:
            write_started = time.perf_counter()
            _rebuild_capacity_prefix(
                adapter,
                sample_id,
                user_id,
                events[: index + 1],
            )
            write_seconds = time.perf_counter() - write_started

        question = _question_text(qa)
        read_started = time.perf_counter()
        records = adapter.retrieve(question, user_id, sample_id, top_k)
        read_seconds = time.perf_counter() - read_started
        options = _options(qa)
        prediction = evaluator.answer_mcq(
            question,
            [record.content for record in records[:top_k]],
            options,
            dataset="membench",
            question_metadata={"agent": sample.get("agent")},
        )
        accuracy = evaluator.score_mcq(
            prediction, str(qa.get("ground_truth") or "")
        )
        sample_results.append(
            {
                "sample_id": sample_id,
                "group": sample.get("group"),
                "question_type": sample.get("question_type"),
                "checkpoint_index": checkpoint_count,
                "message_index": index,
                "tokens": token_count,
                "accuracy": accuracy,
                "prediction": prediction,
                "correct_option": qa.get("ground_truth"),
                "read_time_seconds": read_seconds,
                "write_time_seconds": write_seconds,
                "checkpoint_mode": (
                    "native_incremental" if streaming else "full_prefix_rebuild"
                ),
            }
        )
        checkpoint_count += 1
    if streaming:
        adapter.finalize_capacity_sample()
    return sample_results


def _rebuild_capacity_prefix(
    adapter,
    sample_id: str,
    user_id: str,
    events: list[MemoryEvent],
) -> None:
    """Build one Capacity checkpoint without changing batch-system semantics."""
    adapter.begin_sample(sample_id)
    adapter.reset(user_id)
    for event in events:
        adapter.ingest(event)
    adapter.finalize_ingest()


def _load_statuses(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, dict) else {}


def _load_completed_results(
    path: Path, completed_ids: set[str]
) -> list[dict[str, Any]]:
    if not path.exists() or not completed_ids:
        return []
    results = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if str(item.get("sample_id")) in completed_ids:
            results.append(item)
    return results


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _write_capacity_metrics(
    root: Path,
    results: list[dict[str, Any]],
    statuses: dict[str, dict[str, Any]],
    total_samples: int,
) -> None:
    metrics = summarize_capacity(results, statuses, total_samples)
    _write_json_atomic(root / "capacity_metrics.json", metrics)
    _write_json_atomic(root / "metrics.json", metrics)


def summarize_capacity(
    results: list[dict[str, Any]],
    statuses: dict[str, dict[str, Any]],
    total_samples: int,
) -> dict[str, Any]:
    by_round: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_token_bin: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        by_round[int(item["checkpoint_index"])].append(item)
        by_token_bin[(int(item["tokens"]) // 10_000) * 10_000].append(item)

    def aggregate(groups: dict[int, list[dict[str, Any]]], key_name: str) -> list[dict[str, Any]]:
        curve = []
        for key, items in sorted(groups.items()):
            curve.append(
                {
                    key_name: key,
                    "mean_tokens": sum(int(item["tokens"]) for item in items) / len(items),
                    "accuracy": sum(float(item["accuracy"]) for item in items) / len(items),
                    "observations": len(items),
                }
            )
        return curve

    completed = sum(status.get("status") == "success" for status in statuses.values())
    return {
        "benchmark": "membench",
        "official": {
            "capacity_curve_by_round": aggregate(by_round, "round_after_evidence"),
            "capacity_curve_by_10k_tokens": aggregate(by_token_bin, "token_bin_start"),
        },
        "checkpoints": len(results),
        "total_samples": total_samples,
        "completed_samples": completed,
        "failed_samples": total_samples - completed,
        "sample_coverage": completed / total_samples if total_samples else None,
    }


def _load_system_config(system_context: dict[str, Any] | None) -> dict[str, Any]:
    path = (system_context or {}).get("system_config")
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _capacity_event(sample_id: str, index: int, message: Any) -> MemoryEvent:
    if isinstance(message, dict) and ("user" in message or "agent" in message):
        content = f"{index}[|]'user': {message.get('user', '')}; 'agent': {message.get('agent', '')}"
        role = "user_assistant"
    elif isinstance(message, dict):
        content = f"{index}[|]{message.get('message', '')}"
        role = "user"
    else:
        content = f"{index}[|]{message}"
        role = "user"
    return MemoryEvent(
        event_id=f"{sample_id}-e{index}",
        user_id=sample_id,
        session_id=sample_id,
        role=role,
        content=content,
        timestamp=(
            datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=index)
        ).isoformat(),
        metadata={"step": index},
    )


def _question_text(qa: dict[str, Any]) -> str:
    question = str(qa.get("question") or "")
    current_time = str(qa.get("time") or "").strip()
    return f"{question} (current time is {current_time})" if current_time else question


def _options(qa: dict[str, Any]) -> list[str]:
    choices = qa.get("choices") or {}
    return [f"{letter}. {choices[letter]}" for letter in ("A", "B", "C", "D") if letter in choices]


def _cl100k_tokens(message: Any) -> int:
    global _CL100K_ENCODING
    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError("MemBench capacity requires tiktoken/cl100k_base") from exc
    if _CL100K_ENCODING is None:
        _CL100K_ENCODING = tiktoken.get_encoding("cl100k_base")
    encoding = _CL100K_ENCODING
    if isinstance(message, dict) and ("user" in message or "agent" in message):
        return len(encoding.encode(str(message.get("user") or ""))) + len(
            encoding.encode(str(message.get("agent") or ""))
        )
    if isinstance(message, dict):
        message = message.get("message") or ""
    return len(encoding.encode(str(message)))
