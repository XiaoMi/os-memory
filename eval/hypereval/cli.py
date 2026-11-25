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

from hypereval.registry import check_system, list_systems
from hypereval.runner import compare_runs, run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hypereval", description="Unified benchmark runner for agent memory systems")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("kind", choices=["systems", "datasets"])

    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--system")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--system", required=True)
    run_parser.add_argument("--dataset", required=True)
    run_parser.add_argument("--data-path")
    run_parser.add_argument("--sample-size", type=int)
    run_parser.add_argument("--top-k", type=int, default=10)
    run_parser.add_argument("--output-dir")
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--rerun", action="store_true")
    run_parser.add_argument("--system-config")

    config_parser = subparsers.add_parser("run-config")
    config_parser.add_argument("path")

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("runs", nargs="+")
    compare_parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list":
        if args.kind == "systems":
            for system in list_systems():
                print(f"{system.name}\t{system.status}\t{system.display_name}")
        else:
            print("toy\tready\tDeterministic toy dataset")
            print("locomo\tready\tLoCoMo JSON loader")
            print("longmemeval\tready\tLongMem-Eval JSON/JSONL loader")
            print("gallerymem\tready\tMem-Gallery multimodal JSON loader")
            print("membench\tready\tMemBench trajectory loader")
            print("personamem\tready\tPersonaMem CSV/JSONL loader")
            print("os_events\tready\tOS event SQLite loader")
        return 0
    if args.command == "check":
        systems = [args.system] if args.system else [item.name for item in list_systems()]
        failed = 0
        for name in systems:
            ok, message = check_system(name)
            print(f"{'OK' if ok else 'FAIL'}\t{name}\t{message}")
            failed += not ok
        return int(bool(failed))
    if args.command == "run":
        systems = [item.strip() for item in args.system.split(",") if item.strip()]
        for system in systems:
            output = args.output_dir
            if output and len(systems) > 1:
                output = str(Path(output) / system)
            context = {"system_config": args.system_config} if args.system_config else {}
            root = run_benchmark(system, args.dataset, args.sample_size, output, args.top_k, args.resume, args.rerun, context, args.data_path)
            print(root)
        return 0
    if args.command == "run-config":
        data = json.loads(Path(args.path).read_text(encoding="utf-8"))
        data_path = data.get("data_path")
        for system in data["systems"]:
            if isinstance(system, str):
                system_name = system
                system_output = data.get("output_dir")
                system_context = {}
            else:
                system_name = system["name"]
                system_output = system.get("output_dir", data.get("output_dir"))
                system_context = system.get("context", {})
            root = run_benchmark(system_name, data["dataset"], data.get("sample_size"), system_output, data.get("top_k", 10), data.get("resume", False), data.get("rerun", False), system_context, data_path)
            print(root)
        return 0
    if args.command == "compare":
        print(compare_runs(args.runs, args.output))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
