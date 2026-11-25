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

import copy
import importlib
import importlib.util
import hashlib
import json
import os
import sqlite3
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

from hypereval.adapters.base import MemoryAdapter
from hypereval.model_config import api_key, base_url, model_name
from hypereval.protocol import MemoryEvent, MemoryRecord


def _source_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "systems" / "plugmem" / "source"


class _DeterministicLLMCache:
    """Persist exact temperature-zero PlugMem completions.

    PersonaMem checkpoints are overlapping prefixes. PlugMem must still run its
    complete trajectory structuring pipeline for every checkpoint, but most
    early structuring prompts are byte-for-byte identical. Reusing the exact
    response previously returned for the same model, phase, messages and
    generation parameters preserves that pipeline while avoiding duplicate API
    calls. Non-deterministic calls and empty responses always bypass the cache.
    """

    def __init__(
        self,
        delegate: Any,
        path: Path,
        phase_getter: Callable[[], str],
    ) -> None:
        self._delegate = delegate
        self.path = path
        self._phase_getter = phase_getter
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS completions ("
                "cache_key TEXT PRIMARY KEY, response TEXT NOT NULL)"
            )

    def import_from(self, source_paths: list[Path]) -> tuple[int, int]:
        """Import exact cached responses from prior PlugMem runs.

        Existing target entries always win. If the same exact request key has a
        different stored response, the conflicting source entry is skipped and
        reported instead of silently changing an established evaluation result.
        """
        imported = 0
        conflicts = 0
        target = self.path.resolve()
        with self._lock, self._connect() as destination:
            for source_path in source_paths:
                source = source_path.resolve()
                if source == target or not source.is_file():
                    continue
                try:
                    source_uri = f"file:{source.as_posix()}?mode=ro"
                    with sqlite3.connect(source_uri, uri=True, timeout=30) as origin:
                        rows = origin.execute(
                            "SELECT cache_key, response FROM completions"
                        ).fetchall()
                except (OSError, sqlite3.Error):
                    continue
                for cache_key, response in rows:
                    existing = destination.execute(
                        "SELECT response FROM completions WHERE cache_key = ?",
                        (cache_key,),
                    ).fetchone()
                    if existing is None:
                        cursor = destination.execute(
                            "INSERT OR IGNORE INTO completions(cache_key, response) "
                            "VALUES (?, ?)",
                            (cache_key, response),
                        )
                        if cursor.rowcount == 1:
                            imported += 1
                            continue
                        existing = destination.execute(
                            "SELECT response FROM completions WHERE cache_key = ?",
                            (cache_key,),
                        ).fetchone()
                    if existing is not None and str(existing[0]) != str(response):
                        conflicts += 1
        return imported, conflicts

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _cache_key(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> str:
        payload = {
            "model": str(getattr(self._delegate, "model", "")),
            "phase": self._phase_getter(),
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0,
        top_p: float = 1.0,
        max_tokens: int = 4096,
    ) -> str:
        if float(temperature) != 0.0:
            return self._delegate.complete(
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        cache_key = self._cache_key(messages, temperature, top_p, max_tokens)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT response FROM completions WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if row is not None:
            self._hits += 1
            return str(row[0])

        response = self._delegate.complete(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        self._misses += 1
        if response:
            with self._lock, self._connect() as connection:
                connection.execute(
                    "INSERT OR IGNORE INTO completions(cache_key, response) VALUES (?, ?)",
                    (cache_key, response),
                )
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class PlugMemAdapter(MemoryAdapter):
    """Adapter for PlugMem (TIMAN-group), a graph memory for LLM agents.

    PlugMem 原生是给 agent 轨迹设计的图记忆 (episodic/semantic/procedural 三类
    节点), 官方评测走 FastAPI 服务器。本适配器**纯 in-process** 使用其核心库
    (plugmem 包), 不起服务器: 构造 LLMClient + 本地 bge Embedding + ChromaStorage
    + GraphManager, 每个 LoCoMo 样本一个独立 graph。

    ── PlugMem 写入格式 (参考: 加新数据集时按此转换) ─────────────────────────
    PlugMem 把原始对话变成结构化记忆要走 ``Memory`` structuring 管线:
        mem = Memory(goal, observation, llm, embedder, session_id=...)
        mem.append(action_t0, observation_t1)   # 每一轮调 (多次) LLM 抽取
        mem.close()                              # 收尾, 抽 semantic/procedural
        graph.insert(mem)
    LoCoMo 是 speaker 交替的对话, 我们把每条消息按顺序交替映射成
    (observation, action) 轮次喂给 ``Memory``。**这一步每轮多次 LLM 调用, 成本高**,
    是复现 PlugMem 论文抽取流程的固有代价。

    ── 数据集处理分派 (分发表) ──────────────────────────────────────────
    ``ingest`` 只做缓冲 (PlugMem 需要整段轨迹一起 structuring), 真正写入在
    ``finalize_ingest``。事件 -> (role, content) 由 ``_CONVERTERS`` 按数据集名选
    转换函数, 未登记走 ``_convert_default``。新增数据集若字段不同, 加一个
    ``_convert_<name>`` 方法 + 在 ``_CONVERTERS`` 加一行; 否则什么都不用做。
    """

    # PlugMem 的 semantic 记忆是 LLM 从对话里合成的事实 (facts + tags), 无法从
    # 适配器可见数据可靠回溯到原始 dia_id, 所以 evidence recall/mrr 记为 N/A,
    # 与 SimpleMem 一致。
    supports_evidence_tracing = False
    supports_incremental_checkpoints = True

    def __init__(self) -> None:
        self._graph_manager = None
        self._graph = None
        self._llm = None
        self._embedder = None
        self._context: dict[str, Any] = {}
        self._config: dict[str, Any] = {}
        self._events: dict[str, MemoryEvent] = {}
        self._buffer: list[MemoryEvent] = []
        self._sample_id: str | None = None
        self._dataset: str = ""
        self._top_k: int = 10
        self._llm_cache: _DeterministicLLMCache | None = None
        self._personamem_memory = None
        self._personamem_texts: list[str] = []
        self._personamem_processed_rest = 0

    # ------------------------------------------------------------------
    # preflight: 只检查源码可达, 不触发重依赖 (chromadb/torch) 的实际导入
    # ------------------------------------------------------------------
    @classmethod
    def preflight(cls) -> tuple[bool, str]:
        source_dir = _source_dir()
        package_dir = source_dir / "plugmem"
        if not package_dir.exists():
            return False, f"PlugMem source missing: {package_dir}"
        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))
        if importlib.util.find_spec("plugmem") is None:
            return False, "PlugMem package is not discoverable from the selected source"
        return True, "ready; heavy imports (chromadb/torch) deferred to initialize"

    # ------------------------------------------------------------------
    # initialize: 装配 in-process GraphManager
    # ------------------------------------------------------------------
    def initialize(self, context: dict[str, Any]) -> None:
        self._context = dict(context)
        self._dataset = str(context.get("dataset") or "").lower()
        self._config = self._load_config(context)
        self._top_k = int(self._config.get("top_k", 10))

        source_dir = Path(self._config.get("source_dir", _source_dir())).resolve()
        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))

        try:
            import chromadb
            from plugmem.clients.llm import OpenAICompatibleLLMClient, current_phase
            from plugmem.clients.embedding import PlugMemEmbeddingFunction
            from plugmem.storage.chroma import ChromaStorage
            from plugmem.graph_manager import GraphManager
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "PlugMem runtime cannot be imported (if its source is missing, run `python scripts/fetch_baselines.py` in eval/ first). Install its deps in a dedicated "
                "conda env (see requirements/plugmem.txt). "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc

        # ---- LLM: OpenAI 兼容网关, key/base_url 从 env 读, 不写进 config ----
        model = model_name(self._config, "gpt-4o-mini")
        key = api_key(self._config, "OPENAI_API_KEY", required=True)
        url = base_url(self._config)
        if not url:
            raise RuntimeError(
                "PlugMem needs an OpenAI-compatible base_url. Set OPENAI_BASE_URL "
                "(or model.base_url_env in the config)."
            )
        self._llm = OpenAICompatibleLLMClient(base_url=url, api_key=key, model=model)
        # PlugMem 的 OpenAICompatibleLLMClient 构造裸 ``OpenAI(base_url, api_key)`` 且
        # ``chat.completions.create`` 均不传 timeout, 一次网关无响应即无限 hang
        # (与 A-MEM 同病, 见 README「平台注意事项」)。structuring 每轮多次 LLM 调用,
        # full 长跑迟早撞上网关抖动。适配器层给它的 client 换上超时+重试, 不改源码。
        self._harden_llm_timeout()
        if self._config.get("llm_cache_enabled", True):
            cache_path = self._config.get("llm_cache_path")
            if not cache_path:
                cache_path = Path(self._context["run_dir"]) / "plugmem_llm_cache.sqlite3"
            self._llm_cache = _DeterministicLLMCache(
                self._llm,
                Path(cache_path).resolve(),
                current_phase,
            )
            seed_paths = [
                Path(path)
                for path in self._config.get("llm_cache_seed_paths", [])
                if path
            ]
            imported, conflicts = self._llm_cache.import_from(seed_paths)
            if imported or conflicts:
                print(
                    f"[plugmem] llm cache seed import: imported={imported} "
                    f"conflicts={conflicts}",
                    flush=True,
                )
            self._llm = self._llm_cache
        # stdout 行缓冲在长跑时会让 run.log 长时间 0 字节, 无法判断进程是否卡死;
        # 打一条初始化完成日志并 flush, 给出明确的 "已进入运行" 信号。
        print(
            f"[plugmem] initialized: model={model} top_k={self._top_k} "
            f"llm_timeout={self._config.get('llm_timeout', 60)}s "
            f"llm_max_retries={self._config.get('llm_max_retries', 2)} "
            f"llm_cache={'enabled' if self._llm_cache else 'disabled'}",
            flush=True,
        )

        # ---- Embedding: 本地 bge (自定义 EmbeddingClient 子类), 不走 HTTP ----
        self._embedder = self._build_local_embedder()

        # ---- Chroma: 每次运行持久化到 OUT_DIR/.chroma (可移植, 不污染仓库) ----
        chroma_path = self._config.get("chroma_path")
        if chroma_path:
            chroma_client = chromadb.PersistentClient(path=str(chroma_path))
        else:
            chroma_client = chromadb.EphemeralClient()

        storage = ChromaStorage(
            client=chroma_client,
            embedding_function=PlugMemEmbeddingFunction(self._embedder),
            embedding_client=self._embedder,
        )
        self._graph_manager = GraphManager(
            storage=storage, llm=self._llm, embedder=self._embedder
        )
        self._events = {}
        self._buffer = []
        self._new_graph()

    def _build_local_embedder(self):
        """本地 bge (sentence-transformers) 实现 PlugMem 的 EmbeddingClient 抽象。

        PlugMem 没有内置本地 HF embedder (只有 HTTP / 假 embedding), 这里补一个,
        与 SimpleMem 用同一个本地 bge 模型, 离线可用、不依赖 embedding 端点。
        """
        from plugmem.clients.embedding import EmbeddingClient

        model_path = self._config.get("embedding_model") or os.getenv("PLUGMEM_EMBEDDING_MODEL")
        if not model_path:
            raise RuntimeError(
                "PlugMem embedding model path missing. Set embedding_model in the "
                "config (e.g. {ROOT}/model/bge-base-en-v1.5)."
            )
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "sentence-transformers is required for the local bge embedder. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc

        st_model = SentenceTransformer(str(model_path), device="cpu")
        self_config = self._config

        class _LocalBGEEmbedder(EmbeddingClient):
            def __init__(self, model) -> None:
                self._model = model
                self._cache_size = max(
                    0, int(self_config.get("embedding_cache_size", 32768))
                )
                self._cache = OrderedDict()
                self._cache_lock = threading.Lock()

            def embed(self, text: str) -> list[float]:
                return self.embed_batch([text])[0]

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                if not texts:
                    return []
                normalized = [text or "" for text in texts]
                vectors_by_text = {}
                missing = []
                seen_missing = set()
                with self._cache_lock:
                    for text in normalized:
                        vector = self._cache.get(text)
                        if vector is not None:
                            self._cache.move_to_end(text)
                            vectors_by_text[text] = vector
                        elif text not in seen_missing:
                            missing.append(text)
                            seen_missing.add(text)

                if missing:
                    encoded = self._model.encode(
                        missing,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                    )
                    new_vectors = {
                        text: vector.astype("float32")
                        for text, vector in zip(missing, encoded)
                    }
                    vectors_by_text.update(new_vectors)
                    if self._cache_size:
                        with self._cache_lock:
                            for text, vector in new_vectors.items():
                                self._cache[text] = vector
                                self._cache.move_to_end(text)
                            while len(self._cache) > self._cache_size:
                                self._cache.popitem(last=False)

                return [vectors_by_text[text].tolist() for text in normalized]

        return _LocalBGEEmbedder(st_model)

    def _new_graph(self) -> None:
        """为当前样本建一个隔离的 graph。"""
        graph_id = f"locomo-{self._sample_id or 'default'}"
        if self._graph_manager.storage.graph_exists(graph_id):
            self._graph_manager.delete_graph(graph_id)
        gid = self._graph_manager.create_graph(graph_id)
        self._graph = self._graph_manager.get_graph(gid)

    # ------------------------------------------------------------------
    # 样本生命周期
    # ------------------------------------------------------------------
    def begin_sample(self, sample_id: str) -> None:
        self._sample_id = sample_id
        self._events = {}
        self._buffer = []
        self._personamem_memory = None
        self._personamem_texts = []
        self._personamem_processed_rest = 0
        if self._graph_manager is not None:
            self._new_graph()

    def reset(self, user_id: str) -> None:
        return None

    # ------------------------------------------------------------------
    # ingest: 只缓冲 (PlugMem 需整段轨迹一起 structuring)
    # ------------------------------------------------------------------
    def ingest(self, event: MemoryEvent) -> None:
        self._events[event.event_id] = event
        self._buffer.append(event)

    def _convert_default(self, event: MemoryEvent) -> tuple[str, str]:
        return event.role, event.content

    # Dataset name -> converter producing (role, content). All current datasets
    # use the default; add a _convert_<name> + one line here only if a dataset
    # needs different fields.
    _CONVERTERS: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # finalize_ingest: 走 A 完整 structuring, 把缓冲的对话喂进 Memory 再 insert
    # ------------------------------------------------------------------
    def finalize_ingest(self) -> None:
        if self._graph is None:
            raise RuntimeError("PlugMemAdapter is not initialized")
        if not self._buffer:
            return
        from plugmem.core.memory import Memory

        converter = self._CONVERTERS.get(self._dataset, type(self)._convert_default)
        pending, self._buffer = self._buffer, []
        turns = [converter(self, event) for event in pending]

        if self._dataset == "personamem":
            self._finalize_personamem_checkpoint(Memory, turns)
            return

        goal = self._config.get("memory_goal", "Remember the conversation to answer questions later.")
        session_id = self._sample_id or "session"

        # 把交替对话映射成 (observation, action) 轮次: 第 1 条作为初始 observation,
        # 其后每条轮流作为 action / 下一条 observation。用 "role: content" 保留说话人。
        texts = [f"{role}: {content}" for role, content in turns]
        first_obs = texts[0]
        mem = Memory(
            goal=goal,
            observation=first_obs,
            llm=self._llm,
            embedder=self._embedder,
            session_id=str(session_id),
        )
        # 从第 2 条起, 成对 (action_t0, observation_t1)。奇数条时最后一条用空 observation 收尾。
        rest = texts[1:]
        i = 0
        while i < len(rest):
            action = rest[i]
            observation_next = rest[i + 1] if i + 1 < len(rest) else ""
            mem.append(action_t0=action, observation_t1=observation_next)
            i += 2
        mem.close()
        self._graph.insert(mem)

    def _finalize_personamem_checkpoint(self, memory_class, turns) -> None:
        """Build an exact prefix graph without replaying prior append steps."""
        self._personamem_texts.extend(
            f"{role}: {content}" for role, content in turns
        )
        goal = self._config.get(
            "memory_goal", "Remember the conversation to answer questions later."
        )
        session_id = self._sample_id or "session"
        if self._personamem_memory is None:
            self._personamem_memory = memory_class(
                goal=goal,
                observation=self._personamem_texts[0],
                llm=self._llm,
                embedder=self._embedder,
                session_id=str(session_id),
            )

        rest = self._personamem_texts[1:]
        while self._personamem_processed_rest + 1 < len(rest):
            index = self._personamem_processed_rest
            self._personamem_memory.append(
                action_t0=rest[index],
                observation_t1=rest[index + 1],
            )
            self._personamem_processed_rest += 2

        checkpoint = self._clone_personamem_memory(memory_class, session_id)
        if self._personamem_processed_rest < len(rest):
            checkpoint.append(
                action_t0=rest[self._personamem_processed_rest],
                observation_t1="",
            )
        checkpoint.close()
        self._new_graph()
        self._graph.insert(checkpoint)

    def _clone_personamem_memory(self, memory_class, session_id: str):
        source = self._personamem_memory
        checkpoint = memory_class(
            goal=source.goal,
            observation=source.observation_t0,
            llm=self._llm,
            embedder=self._embedder,
            time=source.time,
            session_id=str(session_id),
        )
        checkpoint.memory = copy.deepcopy(source.memory)
        checkpoint.memory_embedding = copy.deepcopy(source.memory_embedding)
        checkpoint.trajectory = copy.deepcopy(source.trajectory)
        checkpoint.state_t0 = source.state_t0
        checkpoint.observation_t0 = source.observation_t0
        return checkpoint

    # ------------------------------------------------------------------
    # retrieve: 生产检索，自动选择记忆类型并执行 LLM 规划
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 10,
    ) -> list[MemoryRecord]:
        if self._graph is None:
            raise RuntimeError("PlugMemAdapter is not initialized")
        from plugmem.core.value_functions import ProceduralRelevant, SemanticRelevant

        k = top_k or self._top_k
        messages, variables, mode = self._graph.retrieve_memory(
            observation=query,
            goal=self._config.get(
                "memory_goal", "Remember the conversation to answer questions later."
            ),
            subgoal=query,
            state="",
            task_type="question_answering",
            mode=None,
            semantic_relevant=SemanticRelevant(k=k, value_threshold=0.0),
            procedural_relevant=ProceduralRelevant(k=k, value_threshold=0.0),
        )
        records: list[MemoryRecord] = []
        for memory_kind in ("semantic", "procedural", "episodic"):
            content = str(variables.get(f"{memory_kind}_memory") or "").strip()
            if not content or content in {
                "No relevant fact", "No relevant experiences", "No relevant memory"
            }:
                continue
            records.append(
                MemoryRecord(
                    memory_id=f"plugmem-{self._sample_id}-{memory_kind}",
                    content=content,
                    score=None,
                    source_event_ids=(),
                    metadata={
                        "system": "plugmem",
                        "memory_kind": memory_kind,
                        "mode": mode,
                        "retrieval_path": "official_production_retrieval",
                    },
                )
            )
        return records

    def dump_state(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        state: dict[str, Any] = {
            "sample_id": self._sample_id,
            "events": {event_id: event.content for event_id, event in self._events.items()},
            "config": {k: v for k, v in self._config.items() if k != "model"},
        }
        try:
            if self._graph is not None:
                state["semantic_node_count"] = len(self._graph.semantic_id2node)
        except Exception:  # noqa: BLE001
            pass
        if self._llm_cache is not None:
            state["llm_cache_hits"] = self._llm_cache.hits
            state["llm_cache_misses"] = self._llm_cache.misses
            state["llm_cache_path"] = str(self._llm_cache.path)
        (output_dir / "plugmem_adapter.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def close(self) -> None:
        self._graph = None
        self._graph_manager = None
        self._llm = None
        self._embedder = None

    def _harden_llm_timeout(self) -> None:
        """给 PlugMem 的 OpenAI 兼容 client 换上请求级超时 + 有界重试。

        上游 ``OpenAICompatibleLLMClient`` (clients/llm.py) 在 ``__init__`` 里为每个
        base_url 建 ``OpenAI(base_url=..., api_key=...)`` 存进 ``self._clients``, 并用
        ``itertools.cycle`` 轮询; ``complete`` 调 ``chat.completions.create`` 时**不传
        timeout**, 所以一次网关无响应会一直 hang 到 openai SDK 默认 600s。这里就地把
        每个底层 client 用 ``with_options(timeout, max_retries)`` 替换 (共享同一
        auth/base_url, 只覆盖超时与重试), 并重建轮询迭代器。只碰 OpenAI backend, 与
        A-MEM 的 ``_harden_llm_timeout`` 同范式。best-effort: 内部结构不符时静默跳过,
        绝不阻断初始化。

        ``llm_timeout`` (秒, 默认 60) 与 ``llm_max_retries`` (默认 2) 可在系统 config 覆盖。
        """
        import itertools

        timeout = float(self._config.get("llm_timeout", 60))
        max_retries = int(self._config.get("llm_max_retries", 2))
        try:
            clients = getattr(self._llm, "_clients", None)
            if not clients:
                return
            hardened = []
            for client in clients:
                if hasattr(client, "with_options"):
                    hardened.append(
                        client.with_options(timeout=timeout, max_retries=max_retries)
                    )
                else:
                    hardened.append(client)
            self._llm._clients = hardened
            # 轮询迭代器缓存了旧 client 列表, 必须一并重建, 否则 complete() 仍用旧 client。
            self._llm._client_cycle = itertools.cycle(hardened)
            # 上游 complete() 有自己的 max_retries(默认5)+retry_delay(5s) 循环; 我们把它
            # 收紧到与 SDK 一致的 max_retries, 让 "快速失败" 真正快 (否则 5 次 ×(超时+5s) 仍很长)。
            if hasattr(self._llm, "max_retries"):
                self._llm.max_retries = max_retries
        except Exception:  # noqa: BLE001 - 硬化是尽力而为, 绝不阻断运行
            pass

    # ------------------------------------------------------------------
    def _load_config(self, context: dict[str, Any]) -> dict[str, Any]:
        config_path = (
            context.get("plugmem_config")
            or context.get("system_config")
            or os.getenv("PLUGMEM_CONFIG")
        )
        if not config_path:
            return {}
        path = Path(config_path)
        if not path.exists():
            raise RuntimeError(f"PlugMem config does not exist: {path}")
        return json.loads(path.read_text(encoding="utf-8"))
