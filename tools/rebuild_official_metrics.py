#!/usr/bin/env python
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

import argparse
import json
from pathlib import Path
from typing import Any

from hypereval.runner import _summarize


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATUS = ROOT / "runs" / "full_official_matrix" / "parallel_status.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as source:
        for line in source:
            if line.strip():
                results.append(json.loads(line))
    return results


def write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def rebuild_run(run_dir: Path) -> dict[str, Any]:
    config = load_json(run_dir / "config.json")
    prior_metrics = load_json(run_dir / "metrics.json")
    sample_status = load_json(run_dir / "sample_status.json")
    results = load_jsonl(run_dir / "predictions.jsonl")
    metrics = _summarize(
        results,
        sample_status=sample_status,
        total_samples=prior_metrics.get("total_samples"),
        dataset_name=str(config["dataset"]),
    )
    write_json_atomic(run_dir / "metrics.json", metrics)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", default=str(DEFAULT_STATUS))
    args = parser.parse_args()

    status_path = Path(args.status)
    if not status_path.is_absolute():
        status_path = ROOT / status_path
    status = load_json(status_path)

    rebuilt = 0
    for job_name, job in (status.get("jobs") or {}).items():
        if job.get("status") != "completed":
            continue
        run_dir = Path(job["output_dir"])
        metrics = rebuild_run(run_dir)
        categories = len((metrics.get("official") or {}).get("by_category") or {})
        print(
            f"{job_name}: questions={metrics.get('questions', 0)} "
            f"categories={categories}"
        )
        rebuilt += 1

    if rebuilt == 0:
        raise RuntimeError("No completed benchmark runs were found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
