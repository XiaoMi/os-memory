# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""PMM-Bench evaluation entry point.

  python -m pmm_bench users   --data PMM_DIR
  python -m pmm_bench systems
  python -m pmm_bench run     --data PMM_DIR --systems hyperos-mem,mem0 --users all --out runs/pmm
  python -m pmm_bench report  --out runs/pmm
  python -m pmm_bench mini    --data PMM_DIR --users USER --out mini --noise 300

Each (system, user) pair runs in its own process and writes resumable
per-question records under OUT/results/<system>/<user>/. Rerunning the same
command skips finished pairs and retries failed questions.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

from pmm_bench import config as C
from pmm_bench import report as R
from pmm_bench.data import load_questions, select_users

# The line pmm_bench.pmm_benchmark.prepare logs when a resumed run's settings changed.
SETTINGS_CHANGED = '"stage": "settings_changed"'


def finished(out: Path, system: str, user: str, data: Path) -> bool:
    path = out / "results" / system / user / "summary.json"
    if not path.exists():
        return False
    summary = json.loads(path.read_text(encoding="utf-8"))
    overall = summary.get("overall", {})
    return overall.get("completed") == len(load_questions(data / "users" / user)) and summary.get("official", True)


# Baselines and their dependencies (mem0, chromadb, Hugging Face Hub, LiteLLM)
# report usage to third parties by default. Turn that off for every worker;
# a variable already set in the environment wins.
TELEMETRY_OFF = {
    "MEM0_TELEMETRY": "false",
    "ANONYMIZED_TELEMETRY": "False",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "LITELLM_TELEMETRY": "False",
    "DO_NOT_TRACK": "1",
}


def run_pair(system: str, user: str, a, cfg: dict) -> tuple[str, str, str]:
    spec = C.system(cfg, system)
    log = a.out / "logs" / system / f"{user}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    if spec["kind"] == "pending":
        return system, user, "skipped: integration pending (see docs/BASELINES.md)"
    if not a.force and a.limit is None and finished(a.out, system, user, a.data):
        return system, user, "done (cached)"
    python = C.interpreter(cfg, system)
    env = {**TELEMETRY_OFF, **os.environ, "PYTHONPATH": os.pathsep.join([str(C.ROOT), str(C.HYPERMEM_SOURCE), os.environ.get("PYTHONPATH", "")])}
    if spec["kind"] == "hyperos":
        args = json.dumps({"user": user, "data": str(a.data), "out": str(a.out), "config": str(a.config) if a.config else None,
                           "limit": a.limit})
        command = [python, "-c", "import sys,json;from pathlib import Path;from pmm_bench import config as C, hyperos;"
                   "a=json.loads(sys.argv[1]);r=hyperos.run(a['user'],Path(a['data']),Path(a['out']),C.load(a['config']),a['limit'],sys.executable);"
                   "print(json.dumps(r.get('overall',r),ensure_ascii=False))", args]
    else:
        command = [python, "-m", "pmm_bench.worker", "--system", system, "--user", user,
                   "--data", str(a.data), "--out", str(a.out)]
        if a.config:
            command += ["--config", str(a.config)]
        if a.limit is not None:
            command += ["--limit", str(a.limit)]
    start = log.stat().st_size if log.exists() else 0
    with log.open("a", encoding="utf-8") as stream:
        result = subprocess.run(command, env=env, cwd=C.ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        return system, user, f"failed (exit {result.returncode}); see {log}"
    with log.open("rb") as stream:
        stream.seek(start)
        changed = SETTINGS_CHANGED.encode() in stream.read()
    return system, user, f"ok (settings changed since the last run, resumed with the new ones; see {log})" if changed else "ok"


def cmd_run(a) -> int:
    a.out.mkdir(parents=True, exist_ok=True)
    cfg = C.load(a.config)
    if a.input:
        # Workers read settings from a file, so record the override next to the results.
        cfg["input"] = a.input
        a.config = a.out / "config.resolved.json"
        a.config.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    systems = C.select_systems(cfg, a.systems)
    users = select_users(a.data, a.users)
    pairs = [(s, u) for s in systems for u in users]
    print(f"{len(systems)} system(s) x {len(users)} user(s) = {len(pairs)} run(s); output {a.out}", flush=True)
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        for system, user, status in pool.map(lambda p: run_pair(p[0], p[1], a, cfg), pairs):
            print(f"[{system}] {user}: {status}", flush=True)
    report = R.aggregate(a.out, systems, users)
    print((a.out / "report.md").read_text(encoding="utf-8"))
    failed = [s for s, row in report["systems"].items() if row["failed"]]
    return 1 if failed else 0


def cmd_report(a) -> int:
    users = None if a.users in (None, "all") else [u.strip() for u in a.users.split(",")]
    systems = None if a.systems in (None, "all") else [s.strip() for s in a.systems.split(",")]
    R.aggregate(a.out, systems, users)
    print((a.out / "report.md").read_text(encoding="utf-8"))
    return 0


def cmd_users(a) -> int:
    for user in select_users(a.data, "all"):
        print(f"{user}\t{len(load_questions(a.data / 'users' / user))} questions")
    return 0


def cmd_mini(a) -> int:
    from pmm_bench.mini import make
    users = select_users(a.data, a.users)
    for index, user in enumerate(users, 1):
        alias = a.alias if a.alias and len(users) == 1 else (f"{a.alias}{index:02d}" if a.alias else None)
        stats = make(a.data, user, a.out, a.noise, a.seed, alias)
        print(json.dumps({"user": alias or user, **stats}, ensure_ascii=False))
    return 0


def cmd_systems(a) -> int:
    cfg = C.load(a.config)
    for name, spec in cfg["systems"].items():
        print(f"{name:14s} {spec['kind']:8s} {spec.get('display', name)}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="pmm_bench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="build memory and answer questions for systems x users")
    r.add_argument("--data", type=Path, required=True, help="PMM-Bench directory containing users/")
    r.add_argument("--systems", default="hyperos-mem", help="comma list, or all")
    r.add_argument("--users", default="all", help="comma list of user IDs, or all")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--config", type=Path, help="override file merged over configs/pmm/default.json")
    r.add_argument("--input", choices=("cleaned", "raw"), help="observation file to build from (default: cleaned)")
    r.add_argument("--jobs", type=int, default=1, help="(system, user) pairs to run in parallel")
    r.add_argument("--limit", type=int, help="first N observations only; a pipeline check, never an official score")
    r.add_argument("--force", action="store_true", help="rerun pairs that already finished")
    r.set_defaults(func=cmd_run)
    s = sub.add_parser("report", help="aggregate finished results")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--systems")
    s.add_argument("--users")
    s.set_defaults(func=cmd_report)
    u = sub.add_parser("users", help="list users and question counts")
    u.add_argument("--data", type=Path, required=True)
    u.set_defaults(func=cmd_users)
    m = sub.add_parser("mini", help="small package: unchanged QA + evidence records + random noise")
    m.add_argument("--data", type=Path, required=True)
    m.add_argument("--users", required=True, help="comma list of user IDs, or all")
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--noise", type=int, default=300, help="extra non-evidence records per user")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--alias", help="rename the user (several users get alias01, alias02, ...)")
    m.set_defaults(func=cmd_mini)
    y = sub.add_parser("systems", help="list registered systems")
    y.add_argument("--config", type=Path)
    y.set_defaults(func=cmd_systems)
    a = p.parse_args(argv)
    for attr in ("data", "out", "config"):
        if getattr(a, attr, None) is not None:
            setattr(a, attr, getattr(a, attr).resolve())
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
