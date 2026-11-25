# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""HyperOS-Mem V11 benchmark entry point: DRED fast path + SAGE slow path.

    python -m pmm_bench.final describe
    python -m pmm_bench.final run --package PMM_DIR --output OUT --user USER --stage all
    python -m pmm_bench.final check --run OUT/build

``run`` executes the PMM-Bench stages prepare -> dred -> sage -> dashboard ->
evaluate (see pmm_bench.pmm_benchmark) and records final_run_contract.json.
``check`` verifies a finished run read-only, without calling any model.

This is the PMM-Bench orchestration front-end for HyperOS-Mem. To embed the
memory system in your own agent, use ``hypermem.HyperMemory`` instead (see the
repository README.md); that is the supported library interface.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SYSTEM = "HyperOS-Mem"
VERSION = "V11"
METHOD = "DRED + SAGE"
CONTRACT_SCHEMA = "hyperos-mem-final-run-v1"


class FinalContractError(ValueError):
    """The run does not satisfy the HyperOS-Mem V11 completion contract."""


def description() -> dict:
    from pmm_bench.pmm_benchmark import STAGES
    return {
        "schema": "hyperos-mem-final-implementation-v1",
        "system": SYSTEM,
        "version": VERSION,
        "method": METHOD,
        "fast_path": {"name": "DRED", "module": "hypermem.online.engine",
                      "role": "online, LLM-free evidence-deferred event attribution"},
        "slow_path": {"name": "SAGE", "modules": ["hypermem.online.consolidation", "hypermem.online.sage_tools",
                                                  "hypermem.online.sage"],
                      "steps": ["idle boundary consolidation", "ToolSAGE: SplitEvent, SummarizeEvent, ExtractFacts",
                                "LongMemory consolidation"]},
        "retrieval": "hypermem.online.hyperos_retrieval",
        "stages": list(STAGES),
    }


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_final_run(run: str | Path, *, require_sage: bool = True) -> dict:
    """Read-only check that a run completed DRED + SAGE with every gate passed."""
    root = Path(run)
    issues: list[str] = []
    for name in ("config.json", "sage_config.json", "dred_audit.json"):
        if not (root / name).exists():
            issues.append(f"missing {name}")
    if (root / "dred_audit.json").exists() and _load(root / "dred_audit.json").get("status") != "passed":
        issues.append("DRED structural audit did not pass")
    if (root / "sage_config.json").exists() and _load(root / "sage_config.json").get("version") != "hyperos-sage-v1":
        issues.append("SAGE version is not hyperos-sage-v1")
    if require_sage:
        report_path = root / "sage_report.json"
        if not report_path.exists():
            issues.append("missing sage_report.json")
        else:
            report = _load(report_path)
            if report.get("schema") != "hyperos-sage-tools-v1":
                issues.append("sage_report.json was not produced by ToolSAGE")
            if report.get("status") not in ("complete", "complete_with_skipped_failures"):
                issues.append("SAGE completion gate is not complete")
            if report.get("structural_audit", {}).get("status") != "passed":
                issues.append("structural audit after SAGE did not pass")
        if not (root / "frozen.json").exists():
            issues.append("memory was not frozen after SAGE")
    if issues:
        raise FinalContractError("; ".join(issues))
    skipped = _load(root / "sage_report.json").get("skipped_failures", {}) if require_sage else {}
    return {**description(), "run": str(root), "status": "passed", "issues": [],
            "skipped_failures": {k: len(v) for k, v in skipped.items()}}


def _record_contract(run: Path) -> None:
    run.mkdir(parents=True, exist_ok=True)
    value = {"schema": CONTRACT_SCHEMA, "system": SYSTEM, "version": VERSION, "method": METHOD,
             "entrypoint": "pmm_bench.final"}
    path = run / "final_run_contract.json"
    if path.exists() and _load(path) != value:
        raise FinalContractError("Final run contract changed; use a new output directory")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _argument(args: list[str], name: str) -> str | None:
    if name not in args:
        return None
    index = args.index(name)
    if index + 1 >= len(args):
        raise FinalContractError(f"{name} requires a value")
    return args[index + 1]


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["describe"]:
        print(json.dumps(description(), ensure_ascii=False, indent=2))
        return 0
    if args and args[0] == "check":
        parser = argparse.ArgumentParser(prog="pmm_bench.final check")
        parser.add_argument("--run", type=Path, required=True)
        parser.add_argument("--allow-before-sage", action="store_true")
        a = parser.parse_args(args[1:])
        print(json.dumps(validate_final_run(a.run, require_sage=not a.allow_before_sage), ensure_ascii=False, indent=2))
        return 0
    if args and args[0] == "run":
        args.pop(0)
    output, stage = _argument(args, "--output"), _argument(args, "--stage")
    run = Path(output) / ("smoke" if "--smoke" in args else "build") if output else None
    if run is not None and stage in {"evaluate", "dashboard"}:
        validate_final_run(run)
    from pmm_bench import pmm_benchmark
    original = sys.argv
    try:
        sys.argv = [original[0], *args]
        pmm_benchmark.main()
    finally:
        sys.argv = original
    if run is not None:
        _record_contract(run)
        if stage in {"all", "sage"}:
            validate_final_run(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
