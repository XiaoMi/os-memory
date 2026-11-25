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

"""Fetch the baseline systems that PMM-Bench compares against.

The repository does not ship third-party source code. Each
``systems/<name>/manifest.json`` names the upstream repository, the exact
commit that was evaluated and, where needed, a patch with our changes. This
script clones every baseline at that commit into its ``source_dir`` and
applies the patch, giving the same code the published results were run on.

    python scripts/fetch_baselines.py                    # all baselines
    python scripts/fetch_baselines.py --systems mem0,amem
    python scripts/fetch_baselines.py --force            # re-fetch existing copies

Run it from the ``eval/`` directory or anywhere else; paths resolve against
``eval/``. Requires git and network access to the upstream hosts.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = ".pmm_bench_fetched.json"


def manifests() -> dict[str, dict]:
    found = {}
    for path in sorted((ROOT / "systems").glob("*/manifest.json")):
        spec = json.loads(path.read_text(encoding="utf-8"))
        if spec.get("repository") and spec.get("commit") and spec.get("source_dir"):
            found[spec["name"]] = spec
    return found


def git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout.strip()


def expected_marker(spec: dict) -> dict:
    return {"repository": spec["repository"], "commit": spec["commit"], "patches": spec.get("patches", [])}


def fetch(spec: dict, force: bool) -> str:
    target = ROOT / spec["source_dir"]
    marker = target / MARKER
    if target.exists():
        try:
            current = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            current = None
        if current == expected_marker(spec) and not force:
            return "already fetched"
        if not force:
            raise RuntimeError(f"{target} exists but does not match the manifest; rerun with --force to replace it")
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    git("init", "-q", str(target))
    git("remote", "add", "origin", spec["repository"], cwd=target)
    try:
        git("fetch", "-q", "--depth", "1", "origin", spec["commit"], cwd=target)
    except RuntimeError:
        git("fetch", "-q", "origin", cwd=target)  # host refuses fetch-by-commit
    git("checkout", "-q", spec["commit"], cwd=target)
    for patch in spec.get("patches", []):
        git("apply", "--whitespace=nowarn", str(ROOT / patch), cwd=target)
    marker.write_text(json.dumps(expected_marker(spec), indent=2) + "\n", encoding="utf-8")
    return f"fetched {spec['commit'][:12]}" + (f" + {len(spec['patches'])} patch(es)" if spec.get("patches") else "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--systems", default="all", help="comma list of baseline names, or all (default)")
    parser.add_argument("--force", action="store_true", help="replace existing copies")
    args = parser.parse_args(argv)
    available = manifests()
    names = list(available) if args.systems == "all" else [n.strip() for n in args.systems.split(",") if n.strip()]
    unknown = [n for n in names if n not in available]
    if unknown:
        parser.error(f"unknown baseline(s): {', '.join(unknown)}; available: {', '.join(available)}")
    failed = 0
    for name in names:
        try:
            print(f"{name:10s} {fetch(available[name], args.force)}")
        except RuntimeError as exc:
            failed += 1
            print(f"{name:10s} FAILED: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
