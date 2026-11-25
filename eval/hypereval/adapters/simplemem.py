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
import re
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from hypereval.adapters.base import MemoryAdapter, clear_sample_storage
from hypereval.model_config import api_key, base_url, model_name
from hypereval.protocol import MemoryEvent, MemoryRecord, Question


_OMNI_KNOWLEDGE_GRAPH_PATCH_LOCK = threading.RLock()


@contextmanager
def _isolated_omni_knowledge_graph(orchestrator_module: Any, storage_path: Path):
    """Route Omni-SimpleMem's unconfigured knowledge graph into one sample."""
    with _OMNI_KNOWLEDGE_GRAPH_PATCH_LOCK:
        original_knowledge_graph = orchestrator_module.KnowledgeGraph

        def knowledge_graph_factory(*args, **kwargs):
            kwargs.setdefault("storage_path", str(storage_path))
            return original_knowledge_graph(*args, **kwargs)

        orchestrator_module.KnowledgeGraph = knowledge_graph_factory
        try:
            yield
        finally:
            orchestrator_module.KnowledgeGraph = original_knowledge_graph


class SimpleMemAdapter(MemoryAdapter):
    """Adapter for SimpleMem, with Omni-SimpleMem for Mem-Gallery.

    ── SimpleMem accepted data format (参考: 加新数据集时按此转换) ──────────
    ``memory.add_dialogue(role: str, content: str, timestamp: str)``:
        role      = 发言者角色
        content   = 文本内容
        timestamp = 时间戳字符串 (SimpleMem 内部处理, 对格式较宽松)
    嵌入模型由环境变量 EMBEDDING_MODEL 指定 (脚本已设本地 bge)。
    检索: ``hybrid_retriever.retrieve(query)``。

    ── 数据集处理分派 (分发表) ──────────────────────────────────────────
    ``ingest`` 按数据集名从 ``_CONVERTERS`` 选转换函数 (返回 (role, content, timestamp)),
    未登记则走 ``_convert_default``。当前所有数据集都走 default。新增数据集若需要不同的
    字段/时间格式, 加一个 ``_convert_<name>`` 方法 + 在 ``_CONVERTERS`` 加一行; 否则什么都不用做。
    """

    # SimpleMem stores LLM-rewritten "lossless restatements" rather than the raw
    # dialogue, so retrieved memories cannot be reliably traced back to source
    # event ids from adapter-visible data. Evidence recall/mrr are N/A.
    supports_evidence_tracing = False

    def __init__(self) -> None:
        self._memory = None
        self._omni_memory = None
        self._omni_gallery = None
        self._context: dict[str, Any] = {}
        self._config: dict[str, Any] = {}
        self._events: dict[str, MemoryEvent] = {}
        self._sample_id: str | None = None
        self._dataset: str = ""
        self._omni_history_image_count = 0
        self._omni_question_image_count = 0

    @classmethod
    def preflight(cls) -> tuple[bool, str]:
        source_dir = Path(__file__).resolve().parents[2] / "systems" / "simplemem" / "source"
        package_dir = source_dir / "simplemem"
        if not package_dir.exists():
            return False, f"SimpleMem source missing: {package_dir}"
        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))
        if importlib.util.find_spec("simplemem") is None:
            return False, "SimpleMem package is not discoverable from the selected source"
        missing = [
            dependency
            for dependency in ("lance", "tantivy")
            if importlib.util.find_spec(dependency) is None
        ]
        if missing:
            return False, (
                "SimpleMem full keyword retrieval dependencies are missing: "
                + ", ".join(missing)
            )
        return True, "ready; semantic, keyword, and structured retrieval available"

    def initialize(self, context: dict[str, Any]) -> None:
        self._context = dict(context)
        self._dataset = str(context.get("dataset") or "").lower()
        self.supports_evidence_tracing = self._dataset == "gallerymem"
        self._config = self._load_config(context)
        if self._dataset == "gallerymem":
            self._memory = None
            self._omni_memory = None
            self._omni_gallery = None
            self._events = {}
            return
        source_dir = Path(
            self._config.get(
                "source_dir",
                Path(__file__).resolve().parents[2] / "systems" / "simplemem" / "source",
            )
        ).resolve()
        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))
        try:
            module = importlib.import_module("simplemem")
            create = getattr(module, "create")
        except Exception as exc:
            raise RuntimeError(
                "SimpleMem runtime cannot be imported (if its source is missing, run `python scripts/fetch_baselines.py` in eval/ first). Install systems/simplemem/source/setup.py "
                "in a dedicated environment. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        try:
            self._memory = create(mode="text", **self._build_simplemem_config())
        except Exception as exc:
            raise RuntimeError(
                "SimpleMem initialization failed. Check lancedb/pyarrow/tantivy, the embedding "
                "model, the LLM provider, and the database path. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        self._events = {}

    def begin_sample(self, sample_id: str) -> None:
        if self._dataset == "gallerymem":
            if self._omni_gallery is not None:
                self._omni_gallery.orchestrator.close()
            self._sample_id = sample_id
            clear_sample_storage(
                Path(self._context.get("run_dir", "."))
                / "omni_memory_data"
                / sample_id,
                self._context.get("run_dir", "."),
            )
            self._omni_gallery = self._create_omni_gallery(sample_id)
            self._omni_memory = self._omni_gallery.orchestrator
            self._events = {}
            return
        should_reinitialize = self._sample_id is not None and self._memory is not None
        self._sample_id = sample_id
        if should_reinitialize:
            self._reset_text_sample()

    def _reset_text_sample(self) -> None:
        """Clear sample data while keeping SimpleMem's loaded models alive."""
        if self._memory is None:
            return
        vector_store = self._memory.vector_store
        backend = getattr(vector_store, "backend", None)
        table = getattr(backend, "table", None)
        delete = getattr(table, "delete", None)
        if callable(delete):
            delete("entry_id IS NOT NULL")
            backend._fts_initialized = False
        else:
            vector_store.clear()
        builder = self._memory.memory_builder
        builder.dialogue_buffer = []
        builder.processed_count = 0
        builder.previous_entries = []
        self._events = {}

    def reset(self, user_id: str) -> None:
        return None

    def ingest(self, event: MemoryEvent) -> None:
        if self._dataset == "gallerymem":
            if self._omni_gallery is None or self._omni_memory is None:
                raise RuntimeError("Omni-SimpleMem is not initialized")
            self._events[event.event_id] = event
            tags = [
                f"event_id:{event.event_id}",
                f"session_id:{event.session_id}",
            ]
            tags.extend(
                f"image_id:{image_id}"
                for image_id in (event.metadata.get("image_ids") or [])
            )
            image_ids = event.metadata.get("image_ids") or []
            self._omni_gallery.store(
                {
                    "text": event.content,
                    "dialogue_id": event.event_id,
                    "timestamp": event.timestamp,
                    "image": {"img_id": image_ids[0]} if image_ids else None,
                }
            )
            for image_path in event.metadata.get("image_paths") or []:
                if Path(image_path).exists():
                    self._omni_memory.add_image(
                        image_path,
                        session_id=event.session_id,
                        tags=tags,
                    )
                    self._omni_history_image_count += 1
            return
        if self._memory is None:
            raise RuntimeError("SimpleMemAdapter is not initialized")
        self._events[event.event_id] = event
        converter = self._CONVERTERS.get(self._dataset, type(self)._convert_default)
        role, content, timestamp = converter(self, event)
        self._memory.add_dialogue(role, content, timestamp)

    def _convert_default(self, event: MemoryEvent) -> tuple[str, str, str]:
        # SimpleMem takes (role, content, timestamp) directly.
        return event.role, event.content, event.timestamp

    # Dataset name -> converter producing (role, content, timestamp) for
    # add_dialogue. Keyed by the config's _dataset value (lowercased). All current
    # datasets use the default; add a new dataset = add a _convert_<name> method and
    # one line here only if it needs different fields/time handling.
    _CONVERTERS: dict[str, Any] = {}

    def finalize_ingest(self) -> None:
        if self._dataset == "gallerymem":
            if self._omni_memory is not None:
                self._omni_memory.end_session()
            return
        if self._memory is not None:
            self._memory.finalize()

    def retrieve(
        self,
        query: str,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 10,
    ) -> list[MemoryRecord]:
        if self._dataset == "gallerymem":
            return self._retrieve_omni_gallery(query, None, top_k)
        if self._memory is None:
            raise RuntimeError("SimpleMemAdapter is not initialized")
        entries = self._memory.hybrid_retriever.retrieve(query)
        records = []
        for index, entry in enumerate(entries[:top_k]):
            content = str(getattr(entry, "lossless_restatement", entry))
            source_ids = self._match_event_ids(user_id, content, session_id)
            records.append(
                MemoryRecord(
                    memory_id=str(getattr(entry, "entry_id", f"simplemem-{user_id}-{index}")),
                    content=content,
                    score=None,
                    source_event_ids=tuple(source_ids),
                    metadata={
                        "system": "simplemem",
                        "keywords": list(getattr(entry, "keywords", []) or []),
                        "topic": getattr(entry, "topic", None),
                    },
                )
            )
        return records

    def retrieve_question(self, question: Question, top_k: int = 10) -> list[MemoryRecord]:
        """Use Omni-SimpleMem's official Mem-Gallery category-aware recall."""
        if self._dataset != "gallerymem":
            return self.retrieve(
                question.question,
                question.user_id,
                question.session_id,
                top_k,
            )
        query = question.question
        caption = str((question.metadata or {}).get("question_image_caption") or "").strip()
        if caption:
            query += f"\nquestion's image:\nimage_caption: {caption}"
        image_paths = [
            str(path)
            for path in (question.metadata or {}).get("question_image_paths") or []
            if str(path).strip()
        ]
        return self._retrieve_omni_gallery(
            query,
            question.category,
            top_k,
            question_image_paths=image_paths,
        )

    def _retrieve_omni_gallery(
        self,
        query: str,
        category: str | None,
        top_k: int,
        question_image_paths: list[str] | None = None,
    ) -> list[MemoryRecord]:
        if self._omni_gallery is None:
            raise RuntimeError("Omni-SimpleMem is not initialized")
        self._omni_gallery.top_k = max(top_k, 20)
        marked_query = f"[{category}] {query}" if category else query
        if question_image_paths:
            self._omni_question_image_count += len(question_image_paths)
            context = self._omni_gallery.recall_question(
                marked_query,
                question_image_paths,
            )
        else:
            context = self._omni_gallery.recall(marked_query)
        if not context:
            return []
        source_event_ids = self._omni_source_event_ids(context)
        return [
            MemoryRecord(
                memory_id=f"omni-simplemem-{self._sample_id}",
                content=context,
                score=None,
                source_event_ids=tuple(source_event_ids),
                metadata={
                    "system": "omni-simplemem",
                    "category": category,
                    "question_image_paths": list(question_image_paths or []),
                },
            )
        ]

    def _omni_source_event_ids(self, context: str) -> list[str]:
        """Map ordered Omni TURN/IMG annotations back to dataset event IDs."""
        image_events: dict[str, list[str]] = {}
        for event_id, event in self._events.items():
            for image_id in event.metadata.get("image_ids") or []:
                image_events.setdefault(str(image_id), []).append(event_id)

        ordered: list[str] = []
        for line in context.splitlines():
            for turn_id in re.findall(r"TURN:([^|\n]+)", line):
                event_id = turn_id.strip()
                if event_id in self._events and event_id not in ordered:
                    ordered.append(event_id)
            for image_id in re.findall(r"IMG:([^|\s]+)", line):
                for event_id in image_events.get(image_id.strip(), []):
                    if event_id not in ordered:
                        ordered.append(event_id)
        return ordered

    def dump_state(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        backend = (
            getattr(getattr(self._memory, "vector_store", None), "backend", None)
            if self._memory is not None
            else None
        )
        state = {
            "sample_id": self._sample_id,
            "events": {event_id: event.content for event_id, event in self._events.items()},
            "config": self._config,
            "multimodal_backend": (
                "omni-simplemem" if self._dataset == "gallerymem" else None
            ),
            "omni_history_image_count": self._omni_history_image_count,
            "omni_question_image_count": self._omni_question_image_count,
            "text_backend": type(backend).__name__ if backend is not None else None,
            "semantic_retrieval_enabled": callable(
                getattr(backend, "semantic_search", None)
            ),
            "keyword_retrieval_enabled": callable(
                getattr(backend, "keyword_search", None)
            ),
            "structured_retrieval_enabled": callable(
                getattr(backend, "structured_search", None)
            ),
            "fts_initialized": bool(getattr(backend, "_fts_initialized", False)),
        }
        if self._omni_memory is not None and hasattr(self._omni_memory, "get_stats"):
            state["omni_stats"] = self._omni_memory.get_stats()
        if self._memory is not None and hasattr(self._memory, "get_all_memories"):
            state["memory_count"] = len(self._memory.get_all_memories())
        (output_dir / "simplemem_adapter.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    def close(self) -> None:
        if self._omni_gallery is not None:
            self._omni_gallery.orchestrator.close()
        self._omni_gallery = None
        self._omni_memory = None
        if self._memory is not None and hasattr(self._memory, "close"):
            self._memory.close()
        self._memory = None

    def _create_omni_gallery(self, sample_id: str):
        source_dir = Path(
            self._config.get(
                "omni_source_dir",
                Path(__file__).resolve().parents[2]
                / "systems"
                / "simplemem"
                / "source"
                / "OmniSimpleMem",
            )
        ).resolve()
        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))
        try:
            from omni_memory import OmniMemoryConfig
            from benchmarks.memgallery.adapter import OmniMemAdapter
            omni_orchestrator = importlib.import_module("omni_memory.orchestrator")
        except Exception as exc:
            raise RuntimeError(f"Omni-SimpleMem runtime cannot be imported: {exc}") from exc

        config = OmniMemoryConfig.create_default()
        data_dir = Path(self._context.get("run_dir", ".")) / "omni_memory_data" / sample_id
        config.storage.base_dir = str(data_dir)
        config.storage.cold_storage_dir = str(data_dir / "cold_storage")
        config.storage.index_dir = str(data_dir / "index")
        config.embedding.model_name = str(
            self._config.get("omni_embedding_model", "all-MiniLM-L6-v2")
        )
        try:
            config.embedding.embedding_dim = int(
                self._config.get("omni_embedding_dimension", 384)
            )
        except ValueError:
            config.embedding.embedding_dim = 384
        config.embedding.visual_embedding_model = os.getenv(
            "VISUAL_EMBEDDING_MODEL", config.embedding.visual_embedding_model
        )
        config.llm.api_key = api_key(self._config, "OPENAI_API_KEY")
        config.llm.api_base_url = base_url(self._config)
        config.llm.query_model = model_name(self._config, "gpt-4o-mini")
        config.llm.summary_model = config.llm.query_model
        config.llm.caption_model = config.llm.query_model
        config.llm.temperature = 0.0
        config.ensure_directories()
        knowledge_graph_dir = data_dir / "knowledge_graph"

        class IsolatedOmniMemAdapter(OmniMemAdapter):
            def reset(inner_self) -> None:
                with _isolated_omni_knowledge_graph(
                    omni_orchestrator, knowledge_graph_dir
                ):
                    super().reset()

        with _isolated_omni_knowledge_graph(omni_orchestrator, knowledge_graph_dir):
            return IsolatedOmniMemAdapter(data_dir=str(data_dir), config=config)

    def _load_config(self, context: dict[str, Any]) -> dict[str, Any]:
        config_path = context.get("simplemem_config") or context.get("system_config") or os.getenv("SIMPLEMEM_CONFIG")
        if not config_path:
            return {}
        path = Path(config_path)
        if not path.exists():
            raise RuntimeError(f"SimpleMem config does not exist: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _build_simplemem_config(self) -> dict[str, Any]:
        raw_config = dict(self._config.get("simplemem", self._config))
        supported_keys = {
            "api_key",
            "model",
            "base_url",
            "db_path",
            "table_name",
            "clear_db",
            "enable_thinking",
            "use_streaming",
            "enable_planning",
            "enable_reflection",
            "max_reflection_rounds",
            "enable_parallel_processing",
            "max_parallel_workers",
            "enable_parallel_retrieval",
            "max_retrieval_workers",
        }
        config = {
            key: value
            for key, value in raw_config.items()
            if key in supported_keys
        }
        if isinstance(config.get("model"), dict):
            config["model"] = model_name(config, "gpt-4.1-mini")
            config["api_key"] = api_key(config, "OPENAI_API_KEY", required=True)
            configured_base_url = base_url(config)
            if configured_base_url:
                config["base_url"] = configured_base_url
        return config

    def _match_event_ids(self, user_id: str, content: str, session_id: str | None) -> list[str]:
        normalized = content.lower()
        return [
            event_id
            for event_id, event in self._events.items()
            if event.user_id == user_id
            and (not session_id or event.session_id == session_id)
            and (event.content.lower() in normalized or normalized in event.content.lower())
        ]
