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
from pathlib import Path
from typing import Any

from hypereval.adapters.base import MemoryAdapter, clear_sample_storage
from hypereval.image_utils import image_data_url
from hypereval.model_config import api_key, base_url, model_name, model_spec
from hypereval.protocol import MemoryEvent, MemoryRecord


class Mem0Adapter(MemoryAdapter):
    """Adapter for the Mem0 OSS Python SDK using its vector-memory API.

    ── Mem0 accepted data format (参考: 加新数据集时按此转换) ──────────────
    ``memory.add(messages, user_id=..., metadata=..., infer=True)``:
        messages = [{"role": str, "content": str}]   # 单条对话消息
        metadata = {"hypereval_event_id", "session_id", "timestamp"}  # 供检索回溯
    Mem0 不需要显式时间戳解析 (时间只作为 metadata 存储), 故对时间格式不敏感。
    infer=True 时 Mem0 调 LLM 抽取事实; 检索用 search(query, filters={"user_id":...}).

    ── 数据集处理分派 (分发表) ──────────────────────────────────────────
    ``_add_one`` 按数据集名从 ``_CONVERTERS`` 选转换函数, 未登记则走 ``_convert_default``
    (直接用 role/content)。当前所有数据集都走 default。新增数据集若需要不同的消息组织,
    加一个 ``_convert_<name>`` 方法 + 在 ``_CONVERTERS`` 加一行; 否则什么都不用做。
    """

    supports_incremental_checkpoints = True
    supports_capacity_streaming = True

    def __init__(self) -> None:
        self._memory = None
        self._context: dict[str, Any] = {}
        self._config: dict[str, Any] = {}
        self._events: dict[str, MemoryEvent] = {}
        self._pending: list[MemoryEvent] = []
        self._sample_id: str | None = None
        self._sample_storage_path: str | None = None
        self._dataset: str = ""
        self._native_vision_image_count = 0
        self._embedder_cache_installed = False

    @classmethod
    def preflight(cls) -> tuple[bool, str]:
        source_dir = Path(__file__).resolve().parents[2] / "systems" / "mem0" / "source"
        package_dir = source_dir / "mem0"
        if not package_dir.exists():
            return False, f"Mem0 source missing: {package_dir}"
        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))
        if importlib.util.find_spec("mem0") is None:
            return False, "Mem0 package is not discoverable from the selected source"
        return True, "ready; imports deferred"

    def initialize(self, context: dict[str, Any]) -> None:
        self._context = dict(context)
        self._dataset = str(context.get("dataset") or "").lower()
        self._config = self._load_config(context)
        self._require_full_capabilities()
        source_dir = Path(
            self._config.get(
                "source_dir",
                Path(__file__).resolve().parents[2] / "systems" / "mem0" / "source",
            )
        ).resolve()
        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))
        try:
            module = importlib.import_module("mem0")
            memory_class = getattr(module, "Memory")
        except Exception as exc:
            raise RuntimeError(
                "Mem0 runtime cannot be imported (if its source is missing, run `python scripts/fetch_baselines.py` in eval/ first). Install systems/mem0/source/pyproject.toml "
                "in its dedicated environment. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        self._install_embedder_cache()
        try:
            self._memory = memory_class.from_config(self._build_mem0_config())
        except Exception as exc:
            raise RuntimeError(
                "Mem0 initialization failed. Check the LLM, embedding provider, vector-store "
                "path/server, and credentials. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        # Mem0 的 OpenAILLM 也构造裸 ``OpenAI(api_key, base_url)`` 且 ``generate_response``
        # 调 ``chat.completions.create(**params)`` 不传 timeout (OpenAIConfig 无 timeout
        # 字段, config 层无法设), 一次网关无响应即无限 hang (与 A-MEM/PlugMem 同病, 见
        # README「平台注意事项」)。infer=True 时每条消息都调 LLM 抽事实, full 长跑必撞网关
        # 抖动。适配器层给它的 client 换上超时+重试, 不改源码。
        self._harden_llm_timeout()
        print(
            f"[mem0] initialized: infer={self._config.get('infer', True)} "
            f"llm_timeout={self._config.get('llm_timeout', 60)}s "
            f"llm_max_retries={self._config.get('llm_max_retries', 2)}",
            flush=True,
        )
        self._events = {}

    def _install_embedder_cache(self) -> None:
        """Reuse immutable HuggingFace embedders across isolated sample stores.

        Mem0 recreates Memory for every sample so each one gets an isolated
        Qdrant directory. The local embedding model is read-only after loading,
        so rebuilding identical BGE weights for every sample adds cost without
        changing memory behavior. Only the embedder factory result is cached;
        vector stores, history databases, LLM clients, and sample data remain
        newly constructed and isolated.
        """
        try:
            factory_module = importlib.import_module("mem0.utils.factory")
            factory = getattr(factory_module, "EmbedderFactory")
            if getattr(factory, "_hypereval_embedder_cache_installed", False):
                self._embedder_cache_installed = True
                return
            original_create = factory.create
            cache: dict[str, Any] = {}

            def cached_create(cls, provider_name, config, vector_config):
                if str(provider_name).lower() != "huggingface":
                    return original_create(provider_name, config, vector_config)
                key = json.dumps(
                    {"provider": str(provider_name), "config": config},
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
                if key not in cache:
                    cache[key] = original_create(provider_name, config, vector_config)
                return cache[key]

            factory.create = classmethod(cached_create)
            factory._hypereval_embedder_cache_installed = True
            self._embedder_cache_installed = True
        except Exception:  # noqa: BLE001 - optimization must never block a run
            self._embedder_cache_installed = False

    def _require_full_capabilities(self) -> None:
        config = self._config.get("mem0", self._config)
        if self._dataset == "gallerymem":
            llm_config = (config.get("llm") or {}).get("config") or {}
            if llm_config.get("enable_vision") is not True:
                raise RuntimeError(
                    "Mem0 Gallery-Mem requires llm.config.enable_vision=true so "
                    "the original images enter Mem0's native vision pipeline."
                )
        vector_store = config.get("vector_store") or {}
        provider = str(vector_store.get("provider") or "qdrant").lower()
        if provider == "qdrant" and importlib.util.find_spec("fastembed") is None:
            raise RuntimeError(
                "Mem0 full-capability Qdrant retrieval requires fastembed for BM25 "
                "hybrid search. Install requirements/mem0.txt before running benchmarks."
            )

    def begin_sample(self, sample_id: str) -> None:
        self._sample_id = sample_id
        run_dir = Path(self._context.get("run_dir", "."))
        safe_sample = "".join(char if char.isalnum() or char in "-_." else "_" for char in sample_id)
        self._sample_storage_path = str(run_dir / ".qdrant" / safe_sample)
        self._pending = []
        # initialize() runs once before the runner knows the sample id. Recreate
        # for every sample, including the first, so all Qdrant paths are isolated.
        if self._memory is not None:
            self._release_memory()
            clear_sample_storage(
                self._sample_storage_path, self._context.get("run_dir", ".")
            )
            self.initialize(self._context)

    def reset(self, user_id: str) -> None:
        # Mem0 scopes records through user_id. The runner calls begin_sample to
        # isolate samples; reset is intentionally a no-op for a user namespace.
        return None

    def ingest(self, event: MemoryEvent) -> None:
        if self._memory is None:
            raise RuntimeError("Mem0Adapter is not initialized")
        # Queue events; the actual mem0.add() calls (each may hit the LLM for
        # fact extraction) are dispatched serially in finalize_ingest, matching
        # mem0's official usage (the user simply loops over messages calling add).
        self._events[event.event_id] = event
        self._pending.append(event)

    def _add_one(self, event: MemoryEvent) -> None:
        converter = self._CONVERTERS.get(self._dataset, type(self)._convert_default)
        messages = converter(self, event)
        self._memory.add(
            messages,
            user_id=event.user_id,
            metadata={
                "hypereval_event_id": event.event_id,
                "session_id": event.session_id,
                "timestamp": event.timestamp,
            },
            infer=self._config.get("infer", True),
        )

    def _convert_default(self, event: MemoryEvent) -> list[dict[str, Any]]:
        # mem0 accepts standard chat messages; (role, content) is used directly.
        return [{"role": event.role, "content": event.content}]

    def _convert_gallerymem(self, event: MemoryEvent) -> list[dict[str, Any]]:
        image_paths = [
            str(path)
            for path in (event.metadata or {}).get("image_paths") or []
            if str(path).strip()
        ]
        if not image_paths:
            return self._convert_default(event)
        self._native_vision_image_count += len(image_paths)
        content: list[dict[str, Any]] = [{"type": "text", "text": event.content}]
        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": image_data_url(path), "detail": "auto"},
            }
            for path in image_paths
        )
        return [{"role": "user", "content": content}]

    # Dataset name -> converter producing mem0's messages list. Keyed by the
    # config's _dataset value (lowercased). All current datasets use the default;
    # add a new dataset = add a _convert_<name> method and one line here only if
    # that dataset needs a different message shape.
    _CONVERTERS: dict[str, Any] = {"gallerymem": _convert_gallerymem}

    def finalize_ingest(self) -> None:
        if not self._pending:
            return
        # Serial ingestion, mirroring mem0's official flow: one add() per message,
        # in order. No adapter-side rate limiting or worker pool.
        pending, self._pending = self._pending, []
        for event in pending:
            self._add_one(event)

    def retrieve(
        self,
        query: str,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 10,
    ) -> list[MemoryRecord]:
        if self._memory is None:
            raise RuntimeError("Mem0Adapter is not initialized")
        response = self._memory.search(query, filters={"user_id": user_id}, top_k=top_k)
        raw_records = response.get("results", []) if isinstance(response, dict) else response
        records = []
        for index, item in enumerate(raw_records or []):
            if not isinstance(item, dict):
                item = {"memory": str(item)}
            content = str(item.get("memory") or item.get("content") or "")
            source_ids = self._source_ids(item, user_id, content, session_id)
            records.append(
                MemoryRecord(
                    memory_id=str(item.get("id") or f"mem0-{user_id}-{index}"),
                    content=content,
                    score=_float_or_none(item.get("score")),
                    source_event_ids=tuple(source_ids),
                    metadata={key: value for key, value in item.items() if key not in {"memory", "content"}},
                )
            )
        return records

    def dump_state(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "sample_id": self._sample_id,
            "events": {event_id: event.content for event_id, event in self._events.items()},
            "config": self._config,
            "multimodal_backend": (
                "mem0-native-vision" if self._dataset == "gallerymem" else None
            ),
            "native_vision_image_count": self._native_vision_image_count,
            "embedder_cache_installed": self._embedder_cache_installed,
        }
        (output_dir / "mem0_adapter.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    def close(self) -> None:
        self._release_memory()

    def _release_memory(self) -> None:
        vector_store = getattr(self._memory, "vector_store", None)
        client = getattr(vector_store, "client", None)
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 - best effort during store rotation
                pass
        self._memory = None

    def _harden_llm_timeout(self) -> None:
        """给 Mem0 的 OpenAI client 换上请求级超时 + 有界重试。

        上游 ``OpenAILLM`` (llms/openai.py) 构造 ``OpenAI(api_key, base_url)`` 无 timeout,
        ``generate_response`` 调 ``chat.completions.create(**params)`` 也不传 timeout, 且
        ``OpenAIConfig`` 没有 timeout 字段 (config 层无从设置), 故只能在适配器层就地把
        ``self._memory.llm.client`` 用 ``with_options(timeout, max_retries)`` 替换 (共享同一
        auth/base_url, 只覆盖超时与重试)。只碰 OpenAI backend, 与 A-MEM 的
        ``_harden_llm_timeout`` 同范式。best-effort: 内部结构不符 (如非 OpenAI backend)
        时静默跳过, 绝不阻断初始化。

        ``llm_timeout`` (秒, 默认 60) 与 ``llm_max_retries`` (默认 2) 可在系统 config 覆盖。
        """
        timeout = float(self._config.get("llm_timeout", 60))
        max_retries = int(self._config.get("llm_max_retries", 2))
        try:
            llm = getattr(self._memory, "llm", None)
            client = getattr(llm, "client", None)
            if client is None or not hasattr(client, "with_options"):
                return
            llm.client = client.with_options(timeout=timeout, max_retries=max_retries)
        except Exception:  # noqa: BLE001 - 硬化是尽力而为, 绝不阻断运行
            pass

    def _load_config(self, context: dict[str, Any]) -> dict[str, Any]:
        config_path = context.get("mem0_config") or context.get("system_config") or os.getenv("MEM0_CONFIG")
        if not config_path:
            return {}
        path = Path(config_path)
        if not path.exists():
            raise RuntimeError(f"Mem0 config does not exist: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _build_mem0_config(self) -> dict[str, Any]:
        config = dict(self._config.get("mem0", self._config))
        config.pop("source_dir", None)
        config.pop("infer", None)
        spec = model_spec(config)
        if spec:
            llm = dict(config.get("llm", {}))
            llm_config = dict(llm.get("config", {}))
            llm["provider"] = spec.get("provider", llm.get("provider", "openai"))
            llm_config["model"] = model_name(config, llm_config.get("model", "gpt-4o-mini"))
            key = api_key(config, _default_api_key_env(config), required=True)
            if key:
                llm_config["api_key"] = key
            url = base_url(config)
            if url:
                llm_config["openai_base_url"] = url
            llm["config"] = llm_config
            config["llm"] = llm
            config.pop("model", None)
        vector_store = config.get("vector_store")
        if self._sample_storage_path and isinstance(vector_store, dict):
            vector_store = dict(vector_store)
            vector_config = dict(vector_store.get("config") or {})
            vector_config["path"] = self._sample_storage_path
            vector_store["config"] = vector_config
            config["vector_store"] = vector_store
        return config

    def _source_ids(
        self,
        item: dict[str, Any],
        user_id: str,
        content: str,
        session_id: str | None,
    ) -> list[str]:
        metadata = item.get("metadata") or {}
        event_id = metadata.get("hypereval_event_id")
        if event_id in self._events:
            event = self._events[event_id]
            if not session_id or event.session_id == session_id:
                return [event_id]
        normalized = content.lower()
        return [
            event_id
            for event_id, event in self._events.items()
            if event.user_id == user_id
            and (not session_id or event.session_id == session_id)
            and (event.content.lower() in normalized or normalized in event.content.lower())
        ]


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _default_api_key_env(config: dict[str, Any]) -> str:
    provider = str(model_spec(config).get("provider") or config.get("llm", {}).get("provider", "openai")).lower()
    return {
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }.get(provider, "OPENAI_API_KEY")
