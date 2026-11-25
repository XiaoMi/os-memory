# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""Aggregate per-question results into the PMM-Bench table.

Per-type accuracy pools all completed questions of the selected users.
Judge is the unweighted mean of the five per-type accuracies; F1 is the mean
text-overlap F1 over completed questions. Both are reported in percent.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

from pmm_bench.data import CATEGORIES
from pmm_bench.qa import write_json


def _records(directory: Path):
    for path in sorted(directory.glob("*.json")):
        if path.name in {"summary.json", "contract.json"}:
            continue
        yield json.loads(path.read_text(encoding="utf-8"))


def aggregate(out: Path, systems: list[str] | None = None, users: list[str] | None = None) -> dict:
    base = out / "results"
    if systems:
        names = systems
    else:
        names = sorted(p.name for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
    table = {}
    for system in names:
        user_dirs = sorted(p for p in (base / system).iterdir() if p.is_dir()) if (base / system).is_dir() else []
        if users:
            user_dirs = [p for p in user_dirs if p.name in users]
        if not user_dirs:
            continue
        by_type, f1s, per_user, total, failed = defaultdict(list), [], {}, 0, 0
        unofficial = False
        for user_dir in user_dirs:
            summary_path = user_dir / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
            unofficial |= summary.get("official") is False
            done = []
            for record in _records(user_dir):
                total += 1
                if record.get("status") == "complete":
                    done.append(record)
                    by_type[record["category"]].append(record["score"])
                    f1s.append(record["f1"])
                else:
                    failed += 1
            per_user[user_dir.name] = {"completed": len(done),
                                       "accuracy": 100 * sum(r["score"] for r in done) / len(done) if done else None}
        types = {c: (100 * sum(v) / len(v) if (v := by_type.get(c)) else None) for c in CATEGORIES}
        present = [v for v in types.values() if v is not None]
        table[system] = {"users": len(user_dirs), "questions": total, "failed": failed,
                         "by_type": types, "judge": sum(present) / len(present) if len(present) == len(CATEGORIES) else None,
                         "f1": 100 * sum(f1s) / len(f1s) if f1s else None,
                         "official": not unofficial, "per_user": per_user}
    report = {"schema": "pmm-bench-report-v1", "systems": table,
              "judge_definition": "unweighted mean of the five per-type accuracies"}
    write_json(out / "report.json", report)
    (out / "report.md").write_text(markdown(report), encoding="utf-8")
    return report


def _fmt(value):
    return "–" if value is None else f"{value:.1f}"


def markdown(report: dict) -> str:
    lines = ["# PMM-Bench results", "",
             "| Method | Users | " + " | ".join(CATEGORIES) + " | Judge | F1 |",
             "|---|---:|" + "---:|" * (len(CATEGORIES) + 2)]
    for name, row in report["systems"].items():
        flag = "" if row["official"] else " (prefix)"
        cells = [_fmt(row["by_type"][c]) for c in CATEGORIES]
        lines.append(f"| {name}{flag} | {row['users']} | " + " | ".join(cells) + f" | {_fmt(row['judge'])} | {_fmt(row['f1'])} |")
    lines += ["", "Judge = unweighted mean of the five per-type accuracies. Failed questions are excluded and counted below.", ""]
    for name, row in report["systems"].items():
        lines.append(f"## {name}")
        lines.append(f"{row['questions']} question records, {row['failed']} failed.")
        lines.append("")
        lines.append("| User | Completed | Accuracy |")
        lines.append("|---|---:|---:|")
        for user, u in row["per_user"].items():
            lines.append(f"| {user} | {u['completed']} | {_fmt(u['accuracy'])} |")
        lines.append("")
    return "\n".join(lines)
