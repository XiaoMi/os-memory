# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""OpenAI-compatible chat calls for answering and judging (standard library only).

Set PMM_MOCK_LLM=1 to replace every answer/judge call with a deterministic
offline stub. The stub exists for pipeline tests; its scores mean nothing.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request


def normalize_request(request: dict) -> dict:
    """Reasoning models take max_completion_tokens and their default temperature."""
    result = dict(request)
    model = str(result.get("model", ""))
    if re.search(r"(^|/)(gpt-5|o\d)", model):
        result.pop("temperature", None)
        result.pop("top_p", None)
        requested = result.pop("max_tokens", None) or result.get("max_completion_tokens") or 8192
        result["max_completion_tokens"] = max(int(requested), 8192)
    else:
        result["temperature"] = 0.0
    return result


def mocked() -> bool:
    return os.environ.get("PMM_MOCK_LLM") == "1"


def _mock(messages: list[dict]) -> str:
    system = messages[0]["content"]
    user = messages[-1]["content"]
    if "裁判" in system:
        data = json.loads(user)
        hit = any(ref[:4] and ref[:4] in data["prediction"] for ref in data["references"])
        return json.dumps({"label": "CORRECT" if hit else "WRONG", "reason": "mock"}, ensure_ascii=False)
    history = user.split("历史记录：\n", 1)[-1].split("\n\n问题：", 1)[0].strip()
    return history.splitlines()[0][:400] if history else "证据不足，无法回答"


def chat(llm: dict, request: dict) -> dict:
    """POST one chat completion; return {"content", "response"}."""
    request = normalize_request(request)
    if mocked():
        content = _mock(request["messages"])
        return {"content": content, "response": {"mock": True, "choices": [{"message": {"content": content}}]}}
    base = (llm.get("base_url") or os.environ.get(llm.get("base_url_env", "OPENAI_BASE_URL"), "")).rstrip("/")
    key = os.environ.get(llm.get("api_key_env", "OPENAI_API_KEY"), "")
    if not base or not key:
        raise RuntimeError(f"Set {llm.get('base_url_env', 'OPENAI_BASE_URL')} and {llm.get('api_key_env', 'OPENAI_API_KEY')}")
    body = json.dumps(request).encode("utf-8")
    error = None
    for attempt in range(int(llm.get("retries", 2)) + 1):
        try:
            req = urllib.request.Request(base + "/chat/completions", data=body, method="POST",
                                         headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
            with urllib.request.urlopen(req, timeout=float(llm.get("timeout", 180))) as resp:
                response = json.loads(resp.read().decode("utf-8"))
            choice = response["choices"][0]
            if choice.get("finish_reason") == "length":
                raise ValueError("Truncated response")
            content = (choice["message"].get("content") or "").strip()
            if not content:
                raise ValueError("Empty response")
            return {"content": content, "response": response}
        except Exception as exc:  # noqa: BLE001 - retried, then raised
            error = exc
            time.sleep(2 * (attempt + 1))
    raise error
