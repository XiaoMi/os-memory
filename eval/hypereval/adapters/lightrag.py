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
import hashlib
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

from hypereval.adapters.base import MemoryAdapter, clear_sample_storage
from hypereval.image_utils import image_data_url
from hypereval.model_config import api_key, base_url, model_name
from hypereval.protocol import MemoryEvent, MemoryRecord


def _source_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "systems" / "lightrag" / "source"


class LightRAGAdapter(MemoryAdapter):
    """Adapter for LightRAG (HKUDS), a graph-based RAG system.

    LightRAG 是文档型 RAG (不是对话记忆): insert 文本 -> 分块 + LLM 抽取实体/关系建图,
    query 时按 mode 做图/向量检索。本适配器**纯 in-process** 使用其核心库 (默认存储
    JsonKV + nano-vectordb + networkx, 全本地文件, 无需外部服务), LLM 走 OpenAI 兼容网关,
    embedding 用本地 bge (自定义 EmbeddingFunc)。LightRAG 核心是 async, 适配器持有一个长期
    存活的事件循环, 所有 async 操作 (初始化/insert/query) 都在同一个 loop 上跑。

    ── LightRAG 写入格式 (参考: 加新数据集时按此转换) ─────────────────────────
    把整段对话拼成一个文档字符串, 通过 ``ainsert(text)`` 插入 (insert 阶段调 LLM 建图,
    成本高)。检索用 ``aquery_data(question, QueryParam(...))``, 返回结构化 dict,
    ``data["chunks"][*]["content"]`` 即检索到的文本片段。

    ── 数据集处理分派 (分发表) ──────────────────────────────────────────
    ``ingest`` 只缓冲 (LightRAG 需整段文档一起建图), 真正插入在 ``finalize_ingest``。
    事件 -> 一行文本由 ``_CONVERTERS`` 按数据集名选转换函数, 未登记走 ``_convert_default``。
    新增数据集若需不同的行格式, 加一个 ``_convert_<name>`` 方法 + 在 ``_CONVERTERS`` 加一行。
    """

    # LightRAG 检索返回的是分块/实体文本 (LLM 抽取合成), 无法从适配器可见数据回溯到原始
    # 对话事件 id, 所以 evidence recall/mrr 记为 N/A, 与 SimpleMem/PlugMem 一致。
    supports_evidence_tracing = False

    def __init__(self) -> None:
        self._rag = None
        self._context: dict[str, Any] = {}
        self._config: dict[str, Any] = {}
        self._events: dict[str, MemoryEvent] = {}
        self._buffer: list[MemoryEvent] = []
        self._sample_id: str | None = None
        self._dataset: str = ""
        self._top_k: int = 10
        self._mode: str = "mix"
        self._embedder = None  # SentenceTransformer, 复用避免每样本重载
        self._native_vlm_image_count = 0
        self._native_vlm_document_count = 0
        # LightRAG 的存储/worker 内部持有绑定到某个事件循环的 asyncio 队列。必须让
        # 初始化/insert/query 全部跑在**同一个**长期存活的 loop 上——若每步各用一次
        # asyncio.run(每次开关新 loop), 队列会 "bound to a different event loop" 而狂刷错误。
        self._loop = None

    def _run(self, coro):
        """在 adapter 自持的长期事件循环上执行协程 (不新开/关闭 loop)。"""
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)

    # ------------------------------------------------------------------
    # preflight: 只检查源码可达, 不触发重依赖 (torch/nano-vectordb) 的实际导入
    # ------------------------------------------------------------------
    @classmethod
    def preflight(cls) -> tuple[bool, str]:
        source_dir = _source_dir()
        package_dir = source_dir / "lightrag"
        if not package_dir.exists():
            return False, f"LightRAG source missing: {package_dir}"
        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))
        if importlib.util.find_spec("lightrag") is None:
            return False, "LightRAG package is not discoverable from the selected source"
        return True, "ready; heavy imports (torch/nano-vectordb) deferred to initialize"

    # ------------------------------------------------------------------
    def initialize(self, context: dict[str, Any]) -> None:
        self._context = dict(context)
        self._dataset = str(context.get("dataset") or "").lower()
        self._config = self._load_config(context)
        self._top_k = int(self._config.get("top_k", 10))
        self._mode = str(self._config.get("mode", "mix"))
        self._require_full_capabilities()

        source_dir = Path(self._config.get("source_dir", _source_dir())).resolve()
        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))

        # 延迟导入: 校验 LightRAG 运行时可用, 具体实例在 begin_sample 建 (每样本独立目录)
        try:
            importlib.import_module("lightrag")
            importlib.import_module("lightrag.llm.openai")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "LightRAG runtime cannot be imported (if its source is missing, run `python scripts/fetch_baselines.py` in eval/ first). Install its deps in a dedicated "
                "conda env (see requirements/lightrag.txt). "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc

        # 本地 bge embedder (SentenceTransformer), 复用一份
        self._embedder = self._load_bge()
        self._events = {}
        self._buffer = []
        self._new_rag()

    def _require_full_capabilities(self) -> None:
        if self._mode != "mix":
            raise RuntimeError("LightRAG full capability requires mode='mix'.")
        if self._dataset == "gallerymem":
            if self._config.get("vlm_process_enable") is not True:
                raise RuntimeError(
                    "LightRAG Gallery-Mem requires vlm_process_enable=true."
                )
            process_options = str(
                self._config.get("gallerymem_process_options") or ""
            )
            if "i" not in process_options:
                raise RuntimeError(
                    "LightRAG Gallery-Mem requires image analysis in "
                    "gallerymem_process_options."
                )

    def _load_bge(self):
        model_path = self._config.get("embedding_model") or os.getenv("LIGHTRAG_EMBEDDING_MODEL")
        if not model_path:
            raise RuntimeError(
                "LightRAG embedding model path missing. Set embedding_model in the "
                "config (e.g. {ROOT}/model/bge-base-en-v1.5)."
            )
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "sentence-transformers is required for the local bge embedder. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        return SentenceTransformer(str(model_path), device="cpu")

    def _working_dir(self) -> str:
        """每个样本一个独立 working_dir, 隔离 LightRAG 的本地存储 (kv/vdb/graphml)。"""
        base = self._config.get("working_dir") or os.path.join(
            self._context.get("run_dir", "."), ".lightrag"
        )
        return str(Path(base) / self._sample_storage_key())

    def _sample_storage_key(self) -> str:
        sample_id = str(self._sample_id or "default")
        digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:16]
        return f"s-{digest}"

    def _input_dir(self) -> Path:
        base = Path(
            self._config.get("input_dir")
            or Path(self._context.get("run_dir", ".")) / ".lightrag_inputs"
        )
        return (base / self._sample_storage_key()).resolve()

    def _new_rag(self) -> None:
        """构造一个 LightRAG 实例并完成异步初始化 (每样本独立)。"""
        from lightrag import LightRAG
        from lightrag.utils import EmbeddingFunc
        from lightrag.llm.openai import openai_complete_if_cache

        if self._dataset == "gallerymem":
            input_dir = self._input_dir()
            input_dir.mkdir(parents=True, exist_ok=True)
            os.environ["INPUT_DIR"] = str(input_dir)

        model = model_name(self._config, "gpt-4o-mini")
        key = api_key(self._config, "OPENAI_API_KEY", required=True)
        url = base_url(self._config)
        if not url:
            raise RuntimeError(
                "LightRAG needs an OpenAI-compatible base_url. Set OPENAI_BASE_URL "
                "(or model.base_url_env in the config)."
            )

        async def llm_model_func(prompt, system_prompt=None, history_messages=None, keyword_extraction=False, **kwargs):
            return await openai_complete_if_cache(
                model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                base_url=url,
                api_key=key,
                **kwargs,
            )

        embedder = self._embedder
        embedding_dim = int(self._config.get("embedding_dim", 768))

        async def embed_func(texts: list[str]):
            import numpy as np

            def _encode():
                return np.asarray(
                    embedder.encode(list(texts), normalize_embeddings=True), dtype=np.float32
                )

            return await asyncio.to_thread(_encode)

        embedding_func = EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=int(self._config.get("embedding_max_token_size", 8192)),
            func=embed_func,
        )

        async def _build():
            rag = LightRAG(
                working_dir=self._working_dir(),
                llm_model_func=llm_model_func,
                embedding_func=embedding_func,
                vlm_process_enable=bool(
                    self._dataset == "gallerymem"
                    and self._config.get("vlm_process_enable") is True
                ),
            )
            # 本分支的 initialize_storages 已自动包含 initialize_pipeline_status
            await rag.initialize_storages()
            return rag

        self._rag = self._run(_build())

    # ------------------------------------------------------------------
    def begin_sample(self, sample_id: str) -> None:
        self._sample_id = sample_id
        self._events = {}
        self._buffer = []
        if self._embedder is not None:
            self._release_rag()
            clear_sample_storage(
                self._working_dir(), self._context.get("run_dir", ".")
            )
            if self._dataset == "gallerymem":
                clear_sample_storage(
                    self._input_dir(), self._context.get("run_dir", ".")
                )
            self._new_rag()

    def reset(self, user_id: str) -> None:
        return None

    def ingest(self, event: MemoryEvent) -> None:
        self._events[event.event_id] = event
        self._buffer.append(event)

    def _convert_default(self, event: MemoryEvent) -> str:
        return f"{event.role}: {event.content}"

    def _convert_gallerymem(self, event: MemoryEvent) -> str:
        parts = [f"## Memory event {event.event_id}", "", event.content]
        image_ids = list((event.metadata or {}).get("image_ids") or [])
        image_paths = list((event.metadata or {}).get("image_paths") or [])
        for index, raw_path in enumerate(image_paths):
            path = str(raw_path).strip()
            if not path:
                continue
            label = str(image_ids[index]) if index < len(image_ids) else f"image-{index + 1}"
            parts.extend(["", f"![{label}]({image_data_url(path)})"])
            self._native_vlm_image_count += 1
        return "\n".join(parts)

    async def _require_processed_native_document(self) -> None:
        from lightrag.base import DocStatus

        failed_docs = await self._rag.get_docs_by_status(DocStatus.FAILED)
        if failed_docs:
            details = "; ".join(
                f"{doc_id}: {getattr(status, 'error_msg', '') or 'parse failed'}"
                for doc_id, status in failed_docs.items()
            )
            raise RuntimeError(
                f"LightRAG native document processing failed: {details}"
            )
        processed_docs = await self._rag.get_docs_by_status(DocStatus.PROCESSED)
        if not processed_docs:
            raise RuntimeError(
                "LightRAG native document processing produced no processed document."
            )

    # Dataset name -> converter producing one document line. All current datasets use
    # the default; add a _convert_<name> + one line here only if a dataset needs a
    # different line format.
    _CONVERTERS: dict[str, Any] = {"gallerymem": _convert_gallerymem}

    def finalize_ingest(self) -> None:
        if self._rag is None:
            raise RuntimeError("LightRAGAdapter is not initialized")
        if not self._buffer:
            return
        converter = self._CONVERTERS.get(self._dataset, type(self)._convert_default)
        pending, self._buffer = self._buffer, []
        lines = [converter(self, event) for event in pending]
        document = "\n".join(lines)

        async def _insert():
            if self._dataset != "gallerymem":
                await self._rag.ainsert(document)
                return
            from lightrag.constants import (
                FULL_DOCS_FORMAT_PENDING_PARSE,
                PARSER_ENGINE_NATIVE,
            )

            input_dir = self._input_dir()
            input_dir.mkdir(parents=True, exist_ok=True)
            source_path = input_dir / f"{self._sample_storage_key()}.md"
            source_path.write_text(document, encoding="utf-8")
            await self._rag.apipeline_enqueue_documents(
                "",
                file_paths=str(source_path),
                docs_format=FULL_DOCS_FORMAT_PENDING_PARSE,
                parse_engine=PARSER_ENGINE_NATIVE,
                process_options=str(
                    self._config.get("gallerymem_process_options") or "iF"
                ),
            )
            await self._rag.apipeline_process_enqueue_documents()
            await self._require_processed_native_document()
            self._native_vlm_document_count += 1

        self._run(_insert())

    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 10,
    ) -> list[MemoryRecord]:
        if self._rag is None:
            raise RuntimeError("LightRAGAdapter is not initialized")
        from lightrag import QueryParam

        k = top_k or self._top_k

        async def _query():
            # aquery_data 内部强制 only_need_context=True: 只检索、不生成答案,
            # 返回结构化 dict, data["chunks"][*]["content"] 即检索文本片段。
            return await self._rag.aquery_data(
                query, QueryParam(mode=self._mode, top_k=k)
            )

        result = self._run(_query())
        data = result.get("data", {}) if isinstance(result, dict) else {}
        chunks = data.get("chunks", []) or []
        records: list[MemoryRecord] = []
        for index, chunk in enumerate(chunks[:k]):
            content = chunk.get("content", "") if isinstance(chunk, dict) else str(chunk)
            records.append(
                MemoryRecord(
                    memory_id=str(
                        (chunk.get("chunk_id") if isinstance(chunk, dict) else None)
                        or f"lightrag-{self._sample_id}-{index}"
                    ),
                    content=str(content),
                    score=None,
                    source_event_ids=(),  # RAG chunk, 无法回溯原始事件 id
                    metadata={
                        "system": "lightrag",
                        "mode": self._mode,
                        "reference_id": chunk.get("reference_id") if isinstance(chunk, dict) else None,
                    },
                )
            )
        return records

    def dump_state(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "sample_id": self._sample_id,
            "events": {event_id: event.content for event_id, event in self._events.items()},
            "config": {k: v for k, v in self._config.items() if k != "model"},
            "multimodal_backend": (
                "lightrag-native-markdown-vlm"
                if self._dataset == "gallerymem"
                else None
            ),
            "native_vlm_image_count": self._native_vlm_image_count,
            "native_vlm_document_count": self._native_vlm_document_count,
        }
        (output_dir / "lightrag_adapter.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _release_rag(self) -> None:
        """Best-effort teardown of the current LightRAG storages before rebuilding.

        Runs on the same long-lived loop the storages were created on, so their
        bound asyncio queues are torn down cleanly.
        """
        if self._rag is None:
            return
        finalize = getattr(self._rag, "finalize_storages", None)
        if callable(finalize):
            try:
                self._run(finalize())
            except Exception:  # noqa: BLE001 - best effort; never fail on cleanup
                pass
        self._rag = None

    def close(self) -> None:
        self._release_rag()
        self._embedder = None
        # Close the long-lived event loop last, after all storages are finalized.
        # LightRAG spawns background worker tasks on this loop; cancel them, then
        # drain briefly with a timeout so a stuck worker can't hang teardown, then
        # close. Any residual "Task was destroyed" warnings are benign (results are
        # already produced before close()).
        if self._loop is not None:
            try:
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                if pending:
                    self._loop.run_until_complete(
                        asyncio.wait_for(
                            asyncio.gather(*pending, return_exceptions=True),
                            timeout=10,
                        )
                    )
            except Exception:  # noqa: BLE001 - best effort; timeout/cancel noise is fine
                pass
            try:
                self._loop.close()
            except Exception:  # noqa: BLE001 - best effort
                pass
            self._loop = None

    # ------------------------------------------------------------------
    def _load_config(self, context: dict[str, Any]) -> dict[str, Any]:
        config_path = (
            context.get("lightrag_config")
            or context.get("system_config")
            or os.getenv("LIGHTRAG_CONFIG")
        )
        if not config_path:
            return {}
        path = Path(config_path)
        if not path.exists():
            raise RuntimeError(f"LightRAG config does not exist: {path}")
        return json.loads(path.read_text(encoding="utf-8"))
