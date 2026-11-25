# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""Evaluation settings, system registry and per-system interpreters."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "configs" / "pmm" / "default.json"
# HyperOS-Mem's core package (hypermem/) lives at the repository root, outside
# this evaluation tree. Honour PMM_HYPERMEM_SOURCE when set, else default to the
# repository root; fall back to a pip-installed hypermem if neither exists.
HYPERMEM_SOURCE = Path(os.environ.get("PMM_HYPERMEM_SOURCE") or ROOT.parent)


def load(path: str | Path | None = None) -> dict:
    base = json.loads(DEFAULT.read_text(encoding="utf-8"))
    if path and Path(path).resolve() != DEFAULT.resolve():
        override = json.loads(Path(path).read_text(encoding="utf-8"))
        def merge(a, b):
            for k, v in b.items():
                if isinstance(v, dict) and isinstance(a.get(k), dict):
                    merge(a[k], v)
                else:
                    a[k] = v
        merge(base, override)
    return base


def system(cfg: dict, name: str) -> dict:
    systems = cfg["systems"]
    if name not in systems:
        raise ValueError(f"Unknown system {name!r}. Known: {sorted(systems)}")
    return {"name": name, **systems[name]}


def select_systems(cfg: dict, spec: str) -> list[str]:
    names = list(cfg["systems"]) if spec == "all" else [x.strip() for x in spec.split(",") if x.strip()]
    for name in names:
        system(cfg, name)
    return names


def interpreter(cfg: dict, name: str) -> str:
    """Each baseline runs in its own environment: PMM_PYTHON_<NAME> or systems.<name>.python."""
    env = "PMM_PYTHON_" + name.upper().replace("-", "_")
    return os.environ.get(env) or cfg["systems"][name].get("python") or os.environ.get("PMM_PYTHON") or sys.executable


def embedding(cfg: dict) -> dict:
    e = cfg["embedding"]
    url = e.get("url") or os.environ.get(e["url_env"], "")
    model = e.get("model") or os.environ.get(e["model_env"], "")
    return {"url": url, "model": model, "dimension": int(e["dimension"]),
            "batch_size": int(e["batch_size"]), "api_key_env": e.get("api_key_env", "")}


def hyperos_protocol(cfg: dict) -> dict:
    """The protocol file pmm_bench.pmm_benchmark expects, resolved from these settings."""
    emb = embedding(cfg)
    if "/v1/" not in emb["url"]:
        raise ValueError("Set PMM_EMBEDDING_URL to an OpenAI-compatible URL ending in /v1/embeddings")
    base_url = os.environ.get(cfg["llm"]["base_url_env"], "")
    if not base_url:
        raise ValueError(f"Set {cfg['llm']['base_url_env']}")
    return {"configuration": "A", "input_kind": "raw",
            "construction_model": cfg["models"]["construction"],
            "llm_retrieval_model": cfg["models"]["construction"],
            "answer_model": cfg["models"]["answer"], "judge_model": cfg["models"]["judge"],
            "top_k": cfg["retrieval"]["top_k"], "context_chars": cfg["retrieval"]["context_chars"],
            "answer_max_output_tokens": cfg["answer_max_tokens"],
            "judge_max_output_tokens": cfg["judge_max_tokens"],
            "embedding": {"url": emb["url"], "model": emb["model"], "batch_size": emb["batch_size"],
                          "dimension": emb["dimension"], "normalization": "l2", "api_key_env": emb["api_key_env"]},
            "api_base_url": base_url.rstrip("/"), "api_key_env": cfg["llm"]["api_key_env"],
            "abstain": "证据不足，无法回答"}


def adapter_config(cfg: dict, name: str, build_dir: Path) -> dict:
    """Fill a baseline's native config template with models, embedding and paths."""
    spec = system(cfg, name)
    raw = json.loads((ROOT / spec["config"]).read_text(encoding="utf-8"))
    emb = embedding(cfg)
    mapping = {"{ROOT}": ROOT.as_posix(), "{BUILD_DIR}": build_dir.as_posix(),
               "{CONSTRUCTION_MODEL}": cfg["models"]["construction"]}
    def fill(value):
        if isinstance(value, str):
            if value == "{EMBED_DIM}":
                return emb["dimension"]
            for k, v in mapping.items():
                value = value.replace(k, v)
            return value
        if isinstance(value, dict):
            return {k: fill(v) for k, v in value.items()}
        if isinstance(value, list):
            return [fill(v) for v in value]
        return value
    return fill(deepcopy(raw))
