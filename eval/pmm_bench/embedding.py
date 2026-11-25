# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""Shared remote embedding API with a persistent exact-text cache.

Baselines load their encoder through sentence-transformers. Before a baseline
is imported, ``install_api_encoder`` swaps that class for a thin client of the
configured OpenAI-compatible /v1/embeddings endpoint, so every system embeds
with the same model. Set PMM_MOCK_EMBEDDING=1 for an offline hash encoder
(pipeline tests only).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import urllib.request

ENCODER_NAMES = {"api-embedder", "bucketqa-api"}


class APIEncoder:
    def __init__(self, cache_path, config: dict):
        self.config = config
        self.path = str(cache_path)
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        self.namespace = json.dumps({k: config[k] for k in ("url", "model", "dimension")}, sort_keys=True)
        with sqlite3.connect(self.path, timeout=60) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vector BLOB NOT NULL)")

    def key(self, text):
        return hashlib.sha256((self.namespace + "\0" + text).encode()).hexdigest()

    def _request(self, batch):
        import numpy as np
        if os.environ.get("PMM_MOCK_EMBEDDING") == "1":
            out = np.zeros((len(batch), self.config["dimension"]), dtype=np.float32)
            for i, text in enumerate(batch):
                for gram in (text[j:j + 2] for j in range(max(1, len(text) - 1))):
                    out[i, int(hashlib.md5(gram.encode()).hexdigest(), 16) % self.config["dimension"]] += 1
            return out + 1e-3
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.config.get("api_key_env") or "", "")
        if key:
            headers["Authorization"] = "Bearer " + key
        body = json.dumps({"model": self.config["model"], "input": batch}).encode()
        req = urllib.request.Request(self.config["url"], data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=180) as resp:
            rows = json.loads(resp.read().decode())["data"]
        if all("index" in row for row in rows):
            rows = sorted(rows, key=lambda row: row["index"])
        return np.asarray([row["embedding"] for row in rows], dtype=np.float32)

    def encode(self, texts):
        import numpy as np
        texts = list(texts)
        values, missing = {}, []
        with sqlite3.connect(self.path, timeout=60) as db:
            for text in dict.fromkeys(texts):
                row = db.execute("SELECT vector FROM vectors WHERE key=?", (self.key(text),)).fetchone()
                if row:
                    values[text] = np.frombuffer(row[0], dtype=np.float32)
                else:
                    missing.append(text)
        size = self.config["batch_size"]
        for start in range(0, len(missing), size):
            batch = missing[start:start + size]
            for attempt in range(4):
                try:
                    vectors = self._request(batch)
                    if vectors.shape != (len(batch), self.config["dimension"]) or not np.isfinite(vectors).all():
                        raise ValueError(f"Embedding shape/content invalid: {vectors.shape}")
                    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                    if (norms <= 0).any():
                        raise ValueError("Zero embedding vector")
                    vectors = vectors / norms
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(1.5 * (attempt + 1))
            with sqlite3.connect(self.path, timeout=60) as db:
                for text, vector in zip(batch, vectors):
                    db.execute("INSERT OR REPLACE INTO vectors VALUES (?, ?)", (self.key(text), vector.tobytes()))
                    values[text] = vector
        return np.asarray([values[t] for t in texts], dtype=np.float32).reshape(len(texts), self.config["dimension"])


def install_api_encoder(cache_path, config: dict) -> APIEncoder:
    encoder = APIEncoder(cache_path, config)
    import sentence_transformers

    class RemoteSentenceTransformer:
        def __init__(self, model_name_or_path=None, *args, **kwargs):
            if str(model_name_or_path) not in ENCODER_NAMES:
                raise RuntimeError(f"PMM-Bench uses the shared embedding API; unexpected encoder {model_name_or_path!r}")
            self.max_seq_length = 8192
            self.prompts = {}
            self.device = "cpu"

        def get_sentence_embedding_dimension(self):
            return encoder.config["dimension"]

        def encode(self, sentences, *args, **kwargs):
            single = isinstance(sentences, str)
            result = encoder.encode([sentences] if single else sentences)
            if single:
                result = result[0]
            if kwargs.get("convert_to_tensor"):
                import torch
                return torch.from_numpy(result)
            return result

        def to(self, *args, **kwargs):
            return self

        def eval(self):
            return self

    sentence_transformers.SentenceTransformer = RemoteSentenceTransformer
    return encoder
