# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""Run HyperOS-Mem V11 for one user through the benchmark entry point, pmm_bench.final."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from pmm_bench import config as C
from pmm_bench.qa import write_json


def run(user: str, data: Path, out: Path, cfg: dict, limit: int | None = None, python: str | None = None) -> dict:
    """Build and evaluate one user; with ``limit``, build only its first ``limit`` observations and skip QA."""
    if cfg["input"] != "cleaned":
        raise ValueError("OS-Memory's PMM runner builds from the cleaned observations")
    run_root = (out / "runs" / "hyperos-mem" / user).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    # Rewritten on every run so changed settings reach the build; pmm_bench.final
    # reports what changed and refuses a changed embedding model or dimension.
    protocol = run_root / "protocol.resolved.json"
    write_json(protocol, C.hyperos_protocol(cfg))
    env = {**os.environ, "PMM_PROTOCOL": str(protocol),
           "PYTHONPATH": os.pathsep.join([str(C.HYPERMEM_SOURCE), str(C.ROOT), os.environ.get("PYTHONPATH", "")])}
    command = [python or sys.executable, "-m", "pmm_bench.final", "--package", str(data.resolve()),
               "--output", str(run_root), "--user", user, "--stage", "all"]
    if limit is not None:
        command += ["--smoke", "--limit", str(limit)]
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, env=env, cwd=C.ROOT, check=True)
    if limit is not None:
        return {"system": "hyperos-mem", "user": user, "smoke": True,
                "note": f"Prefix build of {limit} observations; OS-Memory skips QA in smoke mode"}
    return collect(user, run_root, out)


def collect(user: str, run_root: Path, out: Path) -> dict:
    """Copy per-question records and the summary into results/hyperos-mem/<user>/."""
    source = run_root / "results" / "A-cleaned" / "hyperos-mem" / user
    destination = out / "results" / "hyperos-mem" / user
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.glob("*.json"):
        if path.name not in {"summary.json", "contract.json"}:
            record = json.loads(path.read_text(encoding="utf-8"))
            record.setdefault("user_id", user)
            write_json(destination / path.name, record)
    shutil.copyfile(source / "contract.json", destination / "contract.json")
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    summary.update(system="hyperos-mem", user=user, input="cleaned", official=True,
                   entrypoint="pmm_bench.final")
    write_json(destination / "summary.json", summary)
    return summary
