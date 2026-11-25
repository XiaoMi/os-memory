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

import importlib
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

from hypereval.adapters.base import MemoryAdapter
from hypereval.model_config import api_key, model_name, model_spec
from hypereval.protocol import MemoryEvent, MemoryRecord


class AMEMAdapter(MemoryAdapter):
    """Adapter for the A-MEM (Agentic Memory) reference implementation.

    ── A-MEM accepted data format (参考: 加新数据集时按此转换) ─────────────
    ``memory.add_note(content: str, time: str)``:
        content = 一段文本 (本适配器拼成 "[user_id] Speaker <role>: <content>")
        time    = 时间戳字符串 (A-MEM 作元数据存储, 对格式不敏感)
    检索: ``search_agentic(query, k=top_k)`` 返回记忆列表。

    ── 数据集处理分派 (分发表) ──────────────────────────────────────────
    ``ingest`` 按数据集名从 ``_CONVERTERS`` 选转换函数 (返回 (content, time)), 未登记则走
    ``_convert_default``。当前所有数据集都走 default。新增数据集若需要不同的 content 拼接/
    时间格式, 加一个 ``_convert_<name>`` 方法 + 在 ``_CONVERTERS`` 加一行; 否则什么都不用做。
    """

    supports_incremental_checkpoints = True
    supports_capacity_streaming = True

    def __init__(self) -> None:
        self._memory = None
        self._context: dict[str, Any] = {}
        self._config: dict[str, Any] = {}
        self._events: dict[str, MemoryEvent] = {}
        self._sample_id: str | None = None
        self._dataset: str = ""

    @classmethod
    def preflight(cls) -> tuple[bool, str]:
        source_dir = Path(__file__).resolve().parents[2] / "systems" / "amem" / "source"
        package_dir = source_dir / "agentic_memory"
        if not package_dir.exists():
            return False, f"A-MEM source missing: {package_dir}"
        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))
        if importlib.util.find_spec("agentic_memory") is None:
            return False, "A-MEM package is not discoverable from the selected source"
        return True, "ready; imports deferred"

    def initialize(self, context: dict[str, Any]) -> None:
        self._context = dict(context)
        self._dataset = str(context.get("dataset") or "").lower()
        self._config = self._load_config(context)
        source_dir = Path(
            self._config.get(
                "source_dir",
                Path(__file__).resolve().parents[2] / "systems" / "amem" / "source",
            )
        ).resolve()
        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))
        try:
            module = importlib.import_module("agentic_memory.memory_system")
            memory_class = getattr(module, "AgenticMemorySystem")
        except Exception as exc:
            raise RuntimeError(
                "A-MEM runtime cannot be imported (if its source is missing, run `python scripts/fetch_baselines.py` in eval/ first). Install systems/amem/source/requirements.txt "
                "in its dedicated environment. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        try:
            self._memory = memory_class(**self._build_amem_config())
        except Exception as exc:
            raise RuntimeError(
                "A-MEM initialization failed. Check the embedding model, LLM backend, "
                "credentials, and Chroma storage configuration. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        # A-MEM's upstream OpenAIController builds ``OpenAI(api_key=...)`` and calls
        # ``chat.completions.create(**kwargs)`` with no timeout, so a single hung
        # gateway request blocks forever (observed: a full run stalled 3.5h on one
        # question). Swap in a client with an aggressive per-request timeout +
        # retries so a stalled call fails fast and recovers instead of hanging.
        self._harden_llm_timeout()
        self._events = {}

    def begin_sample(self, sample_id: str) -> None:
        should_reinitialize = self._sample_id is not None and self._memory is not None
        self._sample_id = sample_id
        if should_reinitialize:
            self.initialize(self._context)

    def reset(self, user_id: str) -> None:
        # A-MEM's reference implementation is collection-scoped rather than
        # user-scoped. User identity is retained in the adapter metadata and
        # event content, while begin_sample isolates benchmark samples.
        return None

    def ingest(self, event: MemoryEvent) -> None:
        if self._memory is None:
            raise RuntimeError("AMEMAdapter is not initialized")
        self._events[event.event_id] = event
        converter = self._CONVERTERS.get(self._dataset, type(self)._convert_default)
        content, time_stamp = converter(self, event)
        self._memory.add_note(content, time=time_stamp)

    def _convert_default(self, event: MemoryEvent) -> tuple[str, str]:
        # A-MEM stores a single text note; embed speaker/user into the content and
        # pass the timestamp through (A-MEM only stores it as metadata).
        content = f"[{event.user_id}] Speaker {event.role}: {event.content}"
        return content, event.timestamp

    # Dataset name -> converter producing (content, time) for add_note. Keyed by the
    # config's _dataset value (lowercased). All current datasets use the default;
    # add a new dataset = add a _convert_<name> method and one line here only if it
    # needs a different content/time shape.
    _CONVERTERS: dict[str, Any] = {}

    def finalize_ingest(self) -> None:
        if self._memory is not None and self._config.get("consolidate", False):
            self._memory.consolidate_memories()

    def retrieve(
        self,
        query: str,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 10,
    ) -> list[MemoryRecord]:
        if self._memory is None:
            raise RuntimeError("AMEMAdapter is not initialized")
        raw_records = self._memory.search_agentic(query, k=top_k)
        records = []
        for index, item in enumerate(raw_records):
            if not isinstance(item, dict):
                item = {"content": str(item)}
            content = str(item.get("content", ""))
            source_ids = self._match_event_ids(user_id, content, session_id)
            records.append(
                MemoryRecord(
                    memory_id=str(item.get("id") or f"amem-{user_id}-{index}"),
                    content=content,
                    score=_float_or_none(item.get("score")),
                    source_event_ids=tuple(source_ids),
                    metadata={key: value for key, value in item.items() if key != "content"},
                )
            )
        return records

    def dump_state(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "sample_id": self._sample_id,
            "events": {event_id: event.content for event_id, event in self._events.items()},
            "config": self._config,
        }
        if self._memory is not None and hasattr(self._memory, "memories"):
            state["memory_ids"] = list(self._memory.memories.keys())
        (output_dir / "amem_adapter.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    def close(self) -> None:
        self._memory = None

    def _harden_llm_timeout(self) -> None:
        """Replace A-MEM's OpenAI client with one that has a request timeout + retries.

        Upstream ``OpenAIController`` (llm_controller.py) constructs
        ``OpenAI(api_key=...)`` without a timeout and calls
        ``chat.completions.create`` with no per-call timeout, so a single
        unresponsive gateway request hangs the whole run indefinitely. We rebuild
        the client in place with the same credentials/base_url but an aggressive
        timeout and bounded retries. Only touches the OpenAI backend; other
        backends (ollama/sglang/openrouter) are left untouched. Best-effort: never
        fails initialization if A-MEM's internals differ from expectations.

        ``llm_timeout`` (seconds, default 60) and ``llm_max_retries`` (default 2)
        can be overridden in the system config.
        """
        timeout = float(self._config.get("llm_timeout", 60))
        max_retries = int(self._config.get("llm_max_retries", 2))
        try:
            controller = getattr(self._memory, "llm_controller", None)
            backend = getattr(controller, "llm", None)
            client = getattr(backend, "client", None)
            # Only the OpenAI backend exposes a `client` that is an OpenAI instance.
            if client is None or not hasattr(client, "with_options"):
                return
            # with_options() returns a shallow copy sharing the same auth/base_url
            # but with overridden timeout/max_retries — no need to re-read secrets.
            backend.client = client.with_options(timeout=timeout, max_retries=max_retries)

            def compatible_get_completion(
                backend_self,
                prompt: str,
                response_format: dict | None = None,
                temperature: float = 1.0,
                max_tokens: int | None = None,
            ) -> str:
                kwargs: dict[str, Any] = {
                    "model": backend_self.model,
                    "messages": [
                        {"role": "system", "content": "You must respond with a JSON object."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                }
                if response_format is not None:
                    kwargs["response_format"] = response_format
                if max_tokens is not None:
                    token_key = (
                        "max_completion_tokens"
                        if any(
                            name in backend_self.model.lower()
                            for name in ("gpt-5", "o1", "o3")
                        )
                        else "max_tokens"
                    )
                    kwargs[token_key] = max_tokens
                attempts = int(self._config.get("llm_content_retries", 5))
                last_error: Exception | None = None
                for attempt in range(attempts):
                    try:
                        response = backend_self.client.chat.completions.create(**kwargs)
                        if isinstance(response, str):
                            content = response.strip()
                        else:
                            choices = getattr(response, "choices", None)
                            if not choices:
                                raise RuntimeError(
                                    "A-MEM gateway response has no completion choices"
                                )
                            content = str(choices[0].message.content or "").strip()
                        normalized = content.lstrip().lower()
                        if not content:
                            raise RuntimeError("A-MEM gateway returned empty content")
                        if normalized.startswith("<!doctype html") or normalized.startswith("<html"):
                            raise RuntimeError("A-MEM gateway returned HTML")
                        return content
                    except Exception as exc:
                        last_error = exc
                        if attempt + 1 < attempts:
                            time.sleep(min(30.0, 2.0 ** attempt))
                raise RuntimeError(
                    f"A-MEM completion failed after {attempts} attempts: {last_error}"
                )

            backend.get_completion = types.MethodType(compatible_get_completion, backend)
        except Exception:  # noqa: BLE001 - hardening is best-effort, never block the run
            pass

    def _load_config(self, context: dict[str, Any]) -> dict[str, Any]:
        config_path = context.get("amem_config") or context.get("system_config") or os.getenv("AMEM_CONFIG")
        if not config_path:
            return {}
        path = Path(config_path)
        if not path.exists():
            raise RuntimeError(f"A-MEM config does not exist: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _build_amem_config(self) -> dict[str, Any]:
        config = dict(self._config.get("amem", self._config))
        config.pop("source_dir", None)
        config.pop("consolidate", None)
        spec = model_spec(config)
        if spec:
            config["llm_backend"] = spec.get("provider", config.get("llm_backend", "openai"))
            config["llm_model"] = model_name(config, config.get("llm_model", "gpt-4o-mini"))
            config["api_key"] = api_key(config, _default_api_key_env(config), required=True)
            config.pop("model", None)
        elif config.get("llm_backend") in {"openai", "openrouter"}:
            config["api_key"] = api_key(config, _default_api_key_env(config), required=True)
        return config

    def _match_event_ids(self, user_id: str, content: str, session_id: str | None) -> list[str]:
        normalized = content.lower()
        matches = []
        for event_id, event in self._events.items():
            if event.user_id != user_id or (session_id and event.session_id != session_id):
                continue
            if event.content.lower() in normalized or normalized in event.content.lower():
                matches.append(event_id)
        return matches


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _default_api_key_env(config: dict[str, Any]) -> str:
    provider = str(model_spec(config).get("provider") or config.get("llm_backend") or "openai").lower()
    return "OPENROUTER_API_KEY" if provider == "openrouter" else "OPENAI_API_KEY"
