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

import asyncio
import os
import threading
import time
from types import SimpleNamespace
from urllib.parse import urlparse


_REQUEST_LOCK = threading.Lock()
_LAST_REQUEST_AT = 0.0


def _wait_for_request_slot() -> None:
    global _LAST_REQUEST_AT
    min_interval = float(os.getenv("HYPEREVAL_LLM_MIN_INTERVAL_SECONDS", "1.0"))
    with _REQUEST_LOCK:
        delay = min_interval - (time.monotonic() - _LAST_REQUEST_AT)
        if delay > 0:
            time.sleep(delay)
        _LAST_REQUEST_AT = time.monotonic()


def _has_chat_completion(response) -> bool:
    return not isinstance(response, str) and bool(getattr(response, "choices", None))


def _has_valid_content(response) -> bool:
    if isinstance(response, str):
        return bool(response.strip()) and not _is_html_response(response)
    return _has_chat_completion(response) and not _is_html_response(response)


def _completion_content(response) -> str:
    try:
        return str(response.choices[0].message.content or "")
    except Exception:
        return ""


def _is_html_response(response) -> bool:
    content = response if isinstance(response, str) else _completion_content(response)
    normalized = str(content).lstrip().lower()
    return normalized.startswith("<!doctype html") or normalized.startswith("<html")


def _normalize_json_content(response):
    try:
        content = response.choices[0].message.content
        if not isinstance(content, str):
            return response
        stripped = content.strip()
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            response.choices[0].message.content = stripped[start : end + 1]
    except Exception:
        pass
    return response


def _coerce_chat_completion(response):
    if not isinstance(response, str):
        return response
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=response))]
    )


def _response_format_unsupported(exc: Exception) -> bool:
    message = str(exc).lower()
    return "response_format" in message and any(
        marker in message for marker in ("unsupported", "not support", "invalid")
    )


def install_openai_gateway_compat(force: bool = False) -> bool:
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or ""
    hostname = (urlparse(base_url).hostname or "").lower()
    # Gateways that reject some OpenAI parameters; extend via HYPEREVAL_COMPAT_HOSTS.
    compatible_hosts = {
        host.strip().lower()
        for host in os.getenv("HYPEREVAL_COMPAT_HOSTS", "").split(",")
        if host.strip()
    }
    if not force and hostname not in compatible_hosts:
        return False

    try:
        from openai.resources.chat.completions import AsyncCompletions, Completions
    except Exception:
        return False

    sync_create = getattr(Completions, "_hypereval_original_create", Completions.create)
    async_create = getattr(
        AsyncCompletions, "_hypereval_original_create", AsyncCompletions.create
    )

    def compatible_sync_create(self, *args, **kwargs):
        requested_format = kwargs.get("response_format")
        attempts = int(os.getenv("HYPEREVAL_LLM_HTML_RETRIES", "5"))
        response = None
        format_downgraded = False
        for attempt in range(attempts):
            _wait_for_request_slot()
            try:
                response = sync_create(self, *args, **kwargs)
            except Exception as exc:
                if requested_format and not format_downgraded and _response_format_unsupported(exc):
                    kwargs = dict(kwargs)
                    kwargs.pop("response_format", None)
                    format_downgraded = True
                    continue
                raise
            if _has_valid_content(response):
                response = _coerce_chat_completion(response)
                return _normalize_json_content(response) if requested_format else response
            if requested_format and not format_downgraded and _is_html_response(response):
                kwargs = dict(kwargs)
                kwargs.pop("response_format", None)
                format_downgraded = True
                continue
            if attempt + 1 < attempts:
                delay = min(30.0, 2.0 ** attempt)
                print(
                    f"[openai-compat] non-JSON gateway response; retrying in {delay:.0f}s",
                    flush=True,
                )
                time.sleep(delay)
        raise RuntimeError("OpenAI-compatible gateway repeatedly returned a non-JSON response")

    async def compatible_async_create(self, *args, **kwargs):
        requested_format = kwargs.get("response_format")
        attempts = int(os.getenv("HYPEREVAL_LLM_HTML_RETRIES", "5"))
        response = None
        format_downgraded = False
        for attempt in range(attempts):
            await asyncio.to_thread(_wait_for_request_slot)
            try:
                response = await async_create(self, *args, **kwargs)
            except Exception as exc:
                if requested_format and not format_downgraded and _response_format_unsupported(exc):
                    kwargs = dict(kwargs)
                    kwargs.pop("response_format", None)
                    format_downgraded = True
                    continue
                raise
            if _has_valid_content(response):
                response = _coerce_chat_completion(response)
                return _normalize_json_content(response) if requested_format else response
            if requested_format and not format_downgraded and _is_html_response(response):
                kwargs = dict(kwargs)
                kwargs.pop("response_format", None)
                format_downgraded = True
                continue
            if attempt + 1 < attempts:
                delay = min(30.0, 2.0 ** attempt)
                print(
                    f"[openai-compat] non-JSON gateway response; retrying in {delay:.0f}s",
                    flush=True,
                )
                await asyncio.sleep(delay)
        raise RuntimeError("OpenAI-compatible gateway repeatedly returned a non-JSON response")

    Completions._hypereval_original_create = sync_create
    AsyncCompletions._hypereval_original_create = async_create
    Completions.create = compatible_sync_create
    AsyncCompletions.create = compatible_async_create
    print("[openai-compat] enabled gateway retries and JSON normalization", flush=True)
    return True
