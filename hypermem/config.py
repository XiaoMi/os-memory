# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""Validated settings for HyperOS-Mem V11 (DRED + SAGE).

``online`` holds the DRED attribution parameters and ``online.consolidation``
the idle boundary consolidation that opens the SAGE slow path. ToolSAGE's
own settings live in ``hypermem.online.sage``.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path

DEFAULT = {
    "embedding": {"backend": "hash", "dimension": 128, "model": "signed-char-ngram-v1", "model_env": "HYPERMEM_EMBEDDING_MODEL", "base_url": "",
                  "base_url_env": "HYPERMEM_EMBEDDING_BASE_URL", "api_key_env": "HYPERMEM_EMBEDDING_API_KEY",
                  "protocol": "openai", "path": "/v1/embeddings", "batch_size": 32, "timeout": 30, "retries": 2},
    "standardization": {"entity_aliases": {}, "max_text_chars": 6000},
    "llm": {"base_url": "", "base_url_env": "HYPERMEM_LLM_BASE_URL", "path": "/v1/chat/completions",
            "api_key_env": "HYPERMEM_LLM_API_KEY", "model": "", "model_env": "HYPERMEM_LLM_MODEL", "timeout": 60, "retries": 2,
            "max_tokens": 512, "temperature": 0},
    "replay": {"progress_every": 100},
    "online": {
        "version": "online-semantic-v1", "mean_mode": "corpus", "fixed_mean": [],
        "tau_a": 0.50, "tau_s": 0.36, "tau_p": 0.60, "tau_c": 0.35,
        "pending_minutes": 3, "active_minutes": 10,
        "alpha_chat": 0.12, "alpha_stream": 0.50, "alpha_default": 0.30,
        "chat_apps": ["微信", "QQ", "com.tencent.mm", "com.tencent.mobileqq"],
        "stream_apps": ["抖音", "小红书", "快手", "com.ss.android.ugc.aweme", "com.xingin.xhs"],
        "split_min_members": 12, "split_min_cluster": 1, "split_max_clusters": 8,
        "split_fit_budget": 512, "split_min_gain": 0.0, "split_iterations": 30,
        "split_enabled": True,
        "split_scope": "all_members",
        "consolidation": {
            "version": "idle-consolidation-v1",
            "ambiguity_margin": 0.08, "candidate_top_k": 8, "batch_size": 1, "batch_candidate_top_k": 3,
            "concurrency": 1,
            "history_minutes": 1440, "stitch_minutes": 10080,
            "key_overlap": 0.6, "archive_days": 30,
            "member_summaries": 6, "text_chars": 1200, "max_attempts": 3,
        },
    },
}


def _positive_int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _unit(value, name, *, closed_low=False):
    low_ok = (lambda v: v >= 0) if closed_low else (lambda v: v > 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not (low_ok(value) and value <= 1):
        raise ValueError(f"{name} must be in {'[0,1]' if closed_low else '(0,1]'}")


def config(overrides: dict | None = None) -> dict:
    result = deepcopy(DEFAULT)

    def merge(target, source, prefix=""):
        for key, value in source.items():
            if key not in target:
                raise ValueError(f"Unknown configuration key: {prefix}{key}")
            if isinstance(target[key], dict):
                if not isinstance(value, dict):
                    raise ValueError(f"{prefix}{key} must be an object")
                if key == "entity_aliases":
                    target[key] = dict(value)
                else:
                    merge(target[key], value, f"{prefix}{key}.")
            else:
                target[key] = value
    merge(result, overrides or {})

    emb, llm, h = result["embedding"], result["llm"], result["online"]
    if emb["backend"] not in {"hash", "http"} or emb["protocol"] not in {"openai", "tei"}:
        raise ValueError("Unsupported embedding backend/protocol")
    for key in ("dimension", "batch_size"):
        _positive_int(emb[key], f"embedding.{key}")
    _positive_int(result["standardization"]["max_text_chars"], "standardization.max_text_chars")
    for section, name in ((emb, "embedding"), (llm, "llm")):
        _positive_int(section["retries"], f"{name}.retries", 0)
        if isinstance(section["timeout"], bool) or not isinstance(section["timeout"], (int, float)) or section["timeout"] <= 0:
            raise ValueError(f"{name}.timeout must be positive")
    _positive_int(llm["max_tokens"], "llm.max_tokens")
    _positive_int(result["replay"]["progress_every"], "replay.progress_every", 0)

    # DRED
    if h["version"] != "online-semantic-v1" or h["mean_mode"] not in {"corpus", "fixed"}:
        raise ValueError("Unsupported DRED version or mean mode")
    if not isinstance(h["split_enabled"], bool):
        raise ValueError("online.split_enabled must be boolean")
    if h["split_scope"] != "all_members":
        raise ValueError("online.split_scope must be all_members")
    for key in ("tau_a", "tau_s", "tau_p", "tau_c", "alpha_chat", "alpha_stream", "alpha_default"):
        _unit(h[key], f"online.{key}")
    if not h["tau_s"] < h["tau_a"] < h["tau_p"]:
        raise ValueError("Online thresholds must satisfy tau_s < tau_a < tau_p")
    for key in ("pending_minutes", "active_minutes"):
        if isinstance(h[key], bool) or not isinstance(h[key], (int, float)) or h[key] <= 0:
            raise ValueError(f"online.{key} must be positive")
    _unit(h["split_min_gain"], "online.split_min_gain", closed_low=True)
    _positive_int(h["split_min_cluster"], "online.split_min_cluster")
    for key in ("split_min_members", "split_max_clusters", "split_fit_budget", "split_iterations"):
        _positive_int(h[key], f"online.{key}", 2)
    for key in ("chat_apps", "stream_apps"):
        if not isinstance(h[key], list) or any(not isinstance(x, str) or not x for x in h[key]):
            raise ValueError(f"online.{key} must be a list of app names")
    from hypermem.schema.serialization import require_embedding
    if h["mean_mode"] == "fixed":
        require_embedding(h["fixed_mean"])
        if not h["fixed_mean"]:
            raise ValueError("fixed centering requires a nonempty fixed_mean")
    elif h["fixed_mean"] != []:
        raise ValueError("fixed_mean only applies to mean_mode=fixed")

    # SAGE boundary consolidation
    c = h["consolidation"]
    if c["version"] != "idle-consolidation-v1":
        raise ValueError("Unsupported consolidation version")
    for key in ("candidate_top_k", "batch_size", "batch_candidate_top_k", "concurrency", "member_summaries", "text_chars", "max_attempts"):
        _positive_int(c[key], f"online.consolidation.{key}")
    for key in ("history_minutes", "stitch_minutes", "archive_days"):
        if isinstance(c[key], bool) or not isinstance(c[key], (int, float)) or not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"online.consolidation.{key} must be positive and finite")
    for key in ("ambiguity_margin", "key_overlap"):
        _unit(c[key], f"online.consolidation.{key}")
    return result


def load_config(path: str | Path | None = None) -> dict:
    return config(json.loads(Path(path).read_text(encoding="utf-8")) if path else None)


def resolve_services(settings):
    settings = deepcopy(settings)
    for service in (settings["embedding"], settings["llm"]):
        for key in ("model", "base_url"):
            if not service[key]:
                service[key] = os.environ.get(service.get(key + "_env", ""), "")
    return settings


# How the embedding service is reached; the rest of the embedding section
# (backend, model, dimension) decides the vectors already stored.
_EMBEDDING_ACCESS = ("base_url", "base_url_env", "api_key_env", "protocol", "path", "batch_size", "timeout", "retries",
                     "model_env")
EMBEDDING_IDENTITY = ("backend", "model", "dimension")


def check_resumable(previous, current, what="settings"):
    """Refuse to resume a memory under settings that would change what is already stored.

    How the model services are reached may change between runs (a corrected
    URL, a new key, another LLM): the whole ``llm`` section and the access
    fields of ``embedding``. A different embedding backend, model or dimension
    makes the stored vectors unusable, and any other setting changes the method.
    """
    if previous is None or previous == current:
        return

    def kept(settings):
        settings = {k: v for k, v in settings.items() if k != "llm"}
        settings["embedding"] = {k: v for k, v in settings.get("embedding", {}).items() if k not in _EMBEDDING_ACCESS}
        return settings

    old, new = kept(previous), kept(current)
    changed = [k for k in EMBEDDING_IDENTITY if old["embedding"].get(k) != new["embedding"].get(k)]
    if changed:
        detail = "; ".join(f"embedding.{k}: {old['embedding'].get(k)!r} -> {new['embedding'].get(k)!r}" for k in changed)
        raise ValueError(f"Embedding model or dimension changed ({detail}); the vectors already stored cannot be "
                         "reused. Build a new memory")
    if old != new:
        raise ValueError(f"Different {what} require a new memory run")
