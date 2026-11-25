# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""Deterministic normalization and derived representations; no event identity inference."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import threading
import unicodedata
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from hypermem.schema.serialization import require_embedding

STRONG_ANCHORS = frozenset({"order_id", "transaction_id", "tracking_id", "reservation_id"})
ANCHOR_PATTERNS = {
    "order_id": r"(?:order[_\s]*(?:id|number|no\.?)|order(?=\s*#)|订单(?:号|编号))\s*[:：#=]?\s*([A-Za-z0-9][A-Za-z0-9_-]{3,})",
    "transaction_id": r"(?:transaction[_\s]*(?:id|no\.?|number)|交易(?:号|编号))\s*[:：#=]?\s*([A-Za-z0-9][A-Za-z0-9_-]{3,})",
    "tracking_id": r"(?:tracking[_\s]*(?:id|no\.?|number)|运单号|快递单号)\s*[:：#=]?\s*([A-Za-z0-9][A-Za-z0-9_-]{3,})",
    "reservation_id": r"(?:reservation[_\s]*(?:id|no\.?|number)|预订号|预定号)\s*[:：#=]?\s*([A-Za-z0-9][A-Za-z0-9_-]{3,})",
    "phone": r"(?<!\d)(?:\+86[- ]?)?(1[3-9]\d{9})(?!\d)",
    "email": r"\b([\w.+-]+@[\w.-]+\.[A-Za-z]{2,})\b",
    "url": r"(https?://[^\s<>\"']+)",
}


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def observation_text(observation, max_chars=6000) -> str:
    try:
        event = json.loads(observation.content)
    except (ValueError, TypeError):
        event = None
    if isinstance(event, dict):
        if observation.provenance.metadata.get("input_format") == "pmm":
            event = event["payload"]["event"]
        data = event.get("data", event)
        parts = [data.get(key) for key in ("title", "content", "text", "text_after", "added_text", "vlm_caption", "description")]
        parts = [part for part in parts if isinstance(part, str) and part.strip()]
        text = "\n".join(dict.fromkeys(parts))
        if not text:
            text = " ".join(str(event.get(key, "")) for key in ("active_name", "active_object"))
    else:
        text = observation.content
    return normalize(text)[:max_chars]


def extract_anchors(text: str, existing: dict | None = None) -> dict[str, list[str]]:
    result = {key: set(values) for key, values in (existing or {}).items()}
    for key, pattern in ANCHOR_PATTERNS.items():
        result.setdefault(key, set()).update(match.rstrip(".,;，。；") for match in re.findall(pattern, text, re.I))
    return {key: sorted(values) for key, values in result.items() if values}


class EntityResolver:
    """Model-independent replaceable interface. Deterministic implementation uses explicit metadata/aliases."""
    def __init__(self, aliases=None):
        self.aliases = {normalize(key).casefold(): normalize(value).casefold() for key, value in (aliases or {}).items()}

    def resolve(self, text: str, explicit=()) -> list[str]:
        values = [value for value in explicit if isinstance(value, str) and value.strip()]
        folded = normalize(text).casefold()
        values.extend(canonical for alias, canonical in self.aliases.items() if alias in folded)
        return sorted({self.aliases.get(normalize(value).casefold(), normalize(value).casefold()) for value in values})


def normalized_vector(vector):
    require_embedding(vector)
    length = math.sqrt(sum(float(value) ** 2 for value in vector))
    return [float(value) / length if length else 0.0 for value in vector]


def cosine(left, right):
    if left is None or right is None:
        return 0.0
    if len(left) != len(right):
        raise ValueError("Embedding dimensions differ; rebuild with a consistent encoder")
    return sum(a * b for a, b in zip(normalized_vector(left), normalized_vector(right)))


class HashEncoder:
    """Reproducible lexical smoke baseline, NOT a trained semantic embedding model."""
    def __init__(self, settings):
        self.settings = settings
        self.calls = 0

    def encode(self, texts):
        vectors = []
        for text in texts:
            text = normalize(text).casefold()
            tokens = re.findall(r"[a-z0-9]+|[\u3400-\u9fff]", text)
            tokens += [text[i:i+2] for i in range(len(text)-1)]
            vector = [0.0] * self.settings["dimension"]
            for token in tokens:
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                vector[int.from_bytes(digest[:4], "big") % len(vector)] += 1 if digest[4] & 1 else -1
            vectors.append(normalized_vector(vector))
        return vectors


_request_log_lock = threading.Lock()


def _log_request(url, payload, attempt, started, response=None, error=None):
    """Opt-in run-local model journal. Authorization headers are never logged."""
    path = os.environ.get("HYPERMEM_REQUEST_LOG")
    if not path:
        return
    embedding = "input" in payload or "inputs" in payload
    record = {"url": url, "model": payload.get("model"), "attempt": attempt + 1,
              "kind": "embedding" if embedding else "chat", "elapsed_seconds": time.monotonic()-started,
              "usage": (response or {}).get("usage") if isinstance(response, dict) else None,
              "error": error}
    if embedding:
        record["input_count"] = len(payload.get("input", payload.get("inputs", [])))
    else:
        record.update(request=payload, response=response)
    with _request_log_lock, open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


class ModelServiceError(RuntimeError):
    """The LLM or embedding service itself failed: unreachable, timing out,
    overloaded, or refusing the configured key, URL or model.

    Unlike a bad reply to one request, this says nothing about the task, so a
    build never spends a task's retry budget on it: the stage stops, its
    journal keeps the task resumable, and the build continues once the service
    is restored.
    """


# HTTP statuses that a retry may clear, and those that mean the key, URL or
# model is wrong. Any other status (400, 413, 422, ...) rejects this request
# only and counts against the task's own budget.
_RETRYABLE_HTTP = frozenset({408, 429, 500, 502, 503, 504})
_SERVICE_HTTP = frozenset({401, 403, 404, 405})


def request_json(settings, payload):
    base = settings.get("base_url", "") or os.environ.get(settings.get("base_url_env", ""))
    if not base:
        raise ValueError("Model service base URL is not configured")
    headers = {"Content-Type": "application/json"}
    key = os.environ.get(settings.get("api_key_env", ""), "")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    url = base.rstrip("/") + "/" + settings["path"].lstrip("/")
    if payload.get("model", "").startswith("azure_openai/"):
        payload = dict(payload)
        payload.pop("chat_template_kwargs", None)  # Local Qwen extension, not an Azure API parameter.
    request = Request(url, data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(), headers=headers, method="POST")
    for attempt in range(settings["retries"] + 1):
        started = time.monotonic()
        try:
            with urlopen(request, timeout=settings["timeout"]) as response:
                result = json.load(response)
            _log_request(url, payload, attempt, started, response=result)
            return result
        # A dropped connection surfaces as ConnectionError or HTTPException, not URLError.
        except (HTTPError, URLError, TimeoutError, ConnectionError, HTTPException) as exc:
            _log_request(url, payload, attempt, started, error={"type": type(exc).__name__, "http_status": getattr(exc, "code", None)})
            if isinstance(exc, HTTPError):
                exc.close()  # an HTTPError holds the open response
            if isinstance(exc, HTTPError) and exc.code in _SERVICE_HTTP:
                raise ModelServiceError(f"Model service at {url} rejected the request: HTTP {exc.code}") from None
            if isinstance(exc, HTTPError) and exc.code not in _RETRYABLE_HTTP:
                raise RuntimeError(f"Model service HTTP {exc.code}") from None
            if attempt == settings["retries"]:
                status = f"HTTP {exc.code}" if isinstance(exc, HTTPError) else type(exc).__name__
                raise ModelServiceError(f"Model service at {url} unavailable after {attempt + 1} attempts ({status})") from None
            time.sleep(min(2 ** attempt, 4))


class HTTPEncoder:
    def __init__(self, settings):
        self.settings = settings
        self.calls = 0
        self.dimension = None

    def encode(self, texts):
        result = []
        if self.settings["protocol"] == "openai" and (not self.settings["model"] or self.settings["model"] == "signed-char-ngram-v1"):
            raise ValueError("Set the actual embedding model name for the OpenAI-compatible service")
        for start in range(0, len(texts), self.settings["batch_size"]):
            batch = texts[start:start + self.settings["batch_size"]]
            payload = {"inputs": batch} if self.settings["protocol"] == "tei" else {"input": batch, "model": self.settings["model"]}
            response = request_json(self.settings, payload)
            self.calls += 1
            if self.settings["protocol"] == "tei":
                vectors = response
            else:
                rows = sorted(response["data"], key=lambda row: row["index"])
                if [row["index"] for row in rows] != list(range(len(batch))):
                    raise ValueError("Embedding response indexes do not match input")
                vectors = [row["embedding"] for row in rows]
            if not isinstance(vectors, list) or len(vectors) != len(batch):
                raise ValueError("Embedding response count does not match input")
            for vector in vectors:
                normalized = normalized_vector(vector)
                if self.dimension is not None and len(vector) != self.dimension:
                    raise ValueError("Embedding service changed dimension")
                self.dimension = len(vector)
                result.append(normalized)
        return result


def make_encoder(settings):
    return HashEncoder(settings) if settings["backend"] == "hash" else HTTPEncoder(settings)
