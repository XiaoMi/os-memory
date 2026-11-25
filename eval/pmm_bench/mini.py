# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""Cut a small runnable package from one PMM-Bench user.

Keeps the user's questions unchanged, keeps every observation that a
question cites as evidence, and adds a fixed number of other observations
sampled at random as noise. The result has the full release layout, so
``python -m pmm_bench run`` and ``pmm_bench.final`` accept it unchanged.

    python -m pmm_bench mini --data PMM-Bench --users USER --out mini --noise 300

The output contains real user records when the input does; keep it local.
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import random
import re

from pmm_bench.data import TZ, file_hash


def evidence_lines(questions: list[dict]) -> set[int]:
    lines = set()
    for q in questions:
        for item in q.get("evidence_event_ids", []):
            match = re.fullmatch(r"(?:raw-)?line:(\d+)", str(item))
            if match:
                lines.add(int(match.group(1)))
    return lines


def make(package: Path, user: str, out: Path, noise: int = 300, seed: int = 0, alias: str | None = None) -> dict:
    package, out = Path(package), Path(out)
    alias = alias or user
    manifest = json.loads((package / "dataset_manifest.json").read_text(encoding="utf-8"))
    info = next((u for u in manifest["users"] if u["user_id"] == user), None)
    if info is None:
        raise ValueError(f"{user} is not in {package / 'dataset_manifest.json'}")
    source = package / "users" / user
    questions = json.loads((source / "qa.json").read_text(encoding="utf-8-sig"))
    wanted = evidence_lines(questions)

    # Cleaned stream: evidence records plus a reproducible random sample of the rest.
    cleaned_rows = (source / "data" / "cleaned_observations.jsonl").read_text(encoding="utf-8-sig").splitlines()
    evidence_idx, other_idx = [], []
    for i, line in enumerate(cleaned_rows):
        lines = set(json.loads(line).get("_cleaning", {}).get("source_lines", []))
        (evidence_idx if lines & wanted else other_idx).append(i)
    noise_idx = random.Random(seed).sample(other_idx, min(noise, len(other_idx)))
    keep = sorted(set(evidence_idx) | set(noise_idx))
    kept_cleaned = [cleaned_rows[i] for i in keep]
    covered = {n for row in kept_cleaned for n in json.loads(row)["_cleaning"]["source_lines"]}

    # Raw stream: the same source lines, so both inputs describe the same history.
    raw_keep = wanted | covered
    raw_rows, raw_found = [], set()
    with (source / "observations.jsonl").open(encoding="utf-8-sig") as stream:
        for line in stream:
            row = json.loads(line)
            number = int(str(row["event_id"]).split(":")[-1])
            if number in raw_keep:
                raw_found.add(number)
                row["user_id"] = alias
                raw_rows.append(json.dumps(row, ensure_ascii=False))

    target = out / "users" / alias
    (target / "data").mkdir(parents=True, exist_ok=True)
    for q in questions:
        q["user_id"] = alias
    (target / "qa.json").write_text(json.dumps(questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (target / "data" / "cleaned_observations.jsonl").write_text("\n".join(kept_cleaned) + "\n", encoding="utf-8")
    (target / "observations.jsonl").write_text("\n".join(raw_rows) + "\n", encoding="utf-8")

    prefix = f"users/{alias}/"
    files = [prefix + f for f in ("qa.json", "observations.jsonl", "data/cleaned_observations.jsonl")]
    checksums = {f: {"sha256": file_hash(out / f), "bytes": (out / f).stat().st_size} for f in files}
    months = sorted({datetime.fromtimestamp(json.loads(r)["ts"] / 1000, TZ).strftime("%Y-%m") for r in kept_cleaned})
    cleaned = prefix + "data/cleaned_observations.jsonl"
    stats = {"source_user": user, "questions": len(questions), "evidence_lines": len(wanted),
             "evidence_lines_in_cleaned": len(wanted & covered), "evidence_lines_in_raw": len(wanted & raw_found),
             "cleaned_records": len(kept_cleaned), "cleaned_evidence_records": len(evidence_idx),
             "cleaned_noise_records": len(set(noise_idx) - set(evidence_idx)), "raw_records": len(raw_rows),
             "noise_seed": seed}
    entry = {"user_id": alias, "questions": prefix + "qa.json", "observations": prefix + "observations.jsonl",
             "event_count": len(raw_rows), "question_count": len(questions),
             "cleaned_observations": cleaned, "cleaned_event_count": len(kept_cleaned),
             "cleaned_sha256": checksums[cleaned]["sha256"],
             "selected_months": sorted(set(info.get("selected_months") or months) | set(months)), "mini": stats}
    out_manifest = {"schema": "pmm-bench-package-v1", "derived_from": "evidence + random noise subset", "users": [entry]}
    if (out / "dataset_manifest.json").exists():
        previous = json.loads((out / "dataset_manifest.json").read_text(encoding="utf-8"))
        out_manifest["users"] = [u for u in previous["users"] if u["user_id"] != alias] + [entry]
        checksums = {**json.loads((out / "checksums.json").read_text(encoding="utf-8")), **checksums}
    (out / "dataset_manifest.json").write_text(json.dumps(out_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "checksums.json").write_text(json.dumps(checksums, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats
