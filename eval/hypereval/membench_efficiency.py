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
from pathlib import Path
from typing import Any

from hypereval.datasets.membench import MemBenchDataset
from hypereval.openai_compat import install_openai_gateway_compat
from hypereval.registry import create_adapter


def run_membench_efficiency(
    system_name: str,
    data_path: str | Path,
    output_dir: str | Path,
    sample_size: int = 40,
    system_context: dict[str, Any] | None = None,
    resume: bool = False,
    retry_rounds: int = 2,
) -> Path:
    install_openai_gateway_compat()
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    samples = MemBenchDataset().load(data_path, sample_size)
    adapter = create_adapter(system_name)
    adapter.initialize(
        {
            "run_dir": str(root),
            "system": system_name,
            "dataset": "membench",
            **(system_context or {}),
        }
    )

    timings_path = root / "write_timings.jsonl"
    status_path = root / "sample_status.json"
    statuses = _load_json(status_path) if resume and status_path.exists() else {}
    completed_ids = {
        sample_id
        for sample_id, status in statuses.items()
        if status.get("status") == "success"
    }
    timings = _load_timings(timings_path, completed_ids) if resume else []
    timings_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in timings),
        encoding="utf-8",
    )

    with timings_path.open("a", encoding="utf-8") as output:
        try:
            for sample in samples:
                if sample.sample_id in completed_ids:
                    continue
                attempts = max(1, int(retry_rounds) + 1)
                for attempt in range(1, attempts + 1):
                    try:
                        adapter.begin_sample(sample.sample_id)
                        for user_id in {event.user_id for event in sample.events}:
                            adapter.reset(user_id)
                        started = time.perf_counter()
                        for event in sample.events:
                            adapter.ingest(event)
                        adapter.finalize_ingest()
                        write_seconds = time.perf_counter() - started
                        item = {
                            "sample_id": sample.sample_id,
                            "group": sample.metadata.get("group"),
                            "events": len(sample.events),
                            "write_seconds": write_seconds,
                            "write_time_seconds": (
                                write_seconds / len(sample.events)
                                if sample.events
                                else None
                            ),
                        }
                        output.write(json.dumps(item, ensure_ascii=False) + "\n")
                        output.flush()
                        timings.append(item)
                        statuses[sample.sample_id] = {
                            "status": "success",
                            "events": len(sample.events),
                            "attempt": attempt,
                        }
                        completed_ids.add(sample.sample_id)
                        break
                    except Exception as exc:  # noqa: BLE001
                        statuses[sample.sample_id] = {
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                            "attempt": attempt,
                        }
                        if attempt == attempts:
                            break
                _write_json_atomic(status_path, statuses)
                _write_metrics(root, timings, statuses, len(samples))
        finally:
            adapter.dump_state(root / "system_state")
            adapter.close()

    _write_json_atomic(status_path, statuses)
    _write_metrics(root, timings, statuses, len(samples))
    return root


def summarize_efficiency(
    timings: list[dict[str, Any]],
    statuses: dict[str, dict[str, Any]],
    total_samples: int,
) -> dict[str, Any]:
    total_events = sum(int(item.get("events") or 0) for item in timings)
    total_write_seconds = sum(float(item.get("write_seconds") or 0.0) for item in timings)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in timings:
        by_group[str(item.get("group") or "unknown")].append(item)

    def operation_time(items: list[dict[str, Any]]) -> float | None:
        events = sum(int(item.get("events") or 0) for item in items)
        if not events:
            return None
        return sum(float(item.get("write_seconds") or 0.0) for item in items) / events

    completed = sum(status.get("status") == "success" for status in statuses.values())
    return {
        "benchmark": "membench",
        "protocol": "write_efficiency_40",
        "official": {
            "overall": {
                "write_time_seconds": (
                    total_write_seconds / total_events if total_events else None
                )
            },
            "by_category": {
                group: {"write_time_seconds": operation_time(items)}
                for group, items in sorted(by_group.items())
            },
        },
        "total_events": total_events,
        "total_samples": total_samples,
        "completed_samples": completed,
        "failed_samples": total_samples - completed,
        "sample_coverage": completed / total_samples if total_samples else None,
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, dict) else {}


def _load_timings(path: Path, completed_ids: set[str]) -> list[dict[str, Any]]:
    if not path.exists() or not completed_ids:
        return []
    return [
        item
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
        for item in [json.loads(line)]
        if str(item.get("sample_id")) in completed_ids
    ]


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _write_metrics(
    root: Path,
    timings: list[dict[str, Any]],
    statuses: dict[str, dict[str, Any]],
    total_samples: int,
) -> None:
    _write_json_atomic(
        root / "metrics.json",
        summarize_efficiency(timings, statuses, total_samples),
    )
