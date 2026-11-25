# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""Build one baseline's memory for one user, then answer and judge its questions.

Runs inside that baseline's own Python environment:
    python -m pmm_bench.worker --system mem0 --user USER --data PMM_DIR --out RUN_DIR
The orchestrator (python -m pmm_bench run) calls this for you.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from pmm_bench import config as C
from pmm_bench.data import load_events, load_questions
from pmm_bench.qa import answer_all, write_json


def run(system: str, user: str, data: Path, out: Path, cfg: dict, limit: int | None = None) -> dict:
    spec = C.system(cfg, system)
    if spec["kind"] != "adapter":
        raise ValueError(f"{system} is not an adapter baseline")
    build_dir = (out / "stores" / system / user).resolve()
    destination = out / "results" / system / user
    build_dir.mkdir(parents=True, exist_ok=True)

    adapter_name = spec.get("adapter", system)
    context = {"run_dir": str(build_dir), "system": adapter_name, "dataset": "pmm"}
    if spec.get("config"):
        native = C.adapter_config(cfg, system, build_dir)
        for key, value in native.get("extra_env", {}).items():
            os.environ[key] = str(value)
        system_config = build_dir / "system_config.json"
        write_json(system_config, native["config"])
        context["system_config"] = str(system_config)
        emb = C.embedding(cfg)
        if os.environ.get("PMM_MOCK_EMBEDDING") != "1" and not (emb["url"] and emb["model"]):
            raise RuntimeError("Set PMM_EMBEDDING_URL and PMM_EMBEDDING_MODEL")
        from pmm_bench.embedding import install_api_encoder
        install_api_encoder(out / "cache" / "embeddings.sqlite3", emb)
        source = native["config"].get("source_dir")
        if source and source not in sys.path:
            sys.path.insert(0, source)

    from hypereval.registry import create_adapter
    events = load_events(data / "users" / user, cfg["input"], limit)
    questions = load_questions(data / "users" / user)
    adapter = create_adapter(adapter_name)
    adapter.initialize(context)
    started = time.monotonic()
    adapter.begin_sample(f"pmm-{user}")
    adapter.reset(user)
    for count, event in enumerate(events, 1):
        adapter.ingest(event)
        if count % 5000 == 0:
            print(json.dumps({"system": system, "user": user, "ingested": count, "total": len(events)}), flush=True)
    adapter.finalize_ingest()
    build = {"system": system, "user": user, "input": cfg["input"], "observations": len(events),
             "prefix_limit": limit, "build_seconds": time.monotonic() - started}
    write_json(build_dir / "build.json", build)

    meta = {"system": system, "user": user, "input": cfg["input"], "prefix_limit": limit,
            "official": limit is None, "models": cfg["models"]}
    try:
        return answer_all(questions, lambda q: adapter.retrieve(q.question, user, None, cfg["retrieval"]["top_k"]),
                          destination, cfg, meta)
    finally:
        adapter.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--system", required=True)
    p.add_argument("--user", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--limit", type=int, help="Use only the first N observations (pipeline check, not an official score)")
    a = p.parse_args(argv)
    summary = run(a.system, a.user, a.data, a.out, C.load(a.config), a.limit)
    print(json.dumps({"system": a.system, "user": a.user, **summary["overall"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
