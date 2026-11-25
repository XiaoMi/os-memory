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

from hypereval.adapters.base import MemoryAdapter
from hypereval.model_config import api_key, base_url, model_name, model_spec
from hypereval.protocol import MemoryEvent, MemoryRecord


class LightMemAdapter(MemoryAdapter):
    """Adapter for the selected LightMem source snapshot.

    LightMem is intentionally imported lazily because its optional runtime has
    heavyweight model/vector dependencies. The benchmark core and Mock adapter
    remain usable when that environment is not installed.

    ── LightMem accepted data format (参考: 加新数据集时按此转换) ─────────────
    ``add_memory([message], force_segment=True, force_extract=True)`` 每条 message 是一个 dict:
        {
          "role":       str,   # 发言者角色 (user / assistant / speaker 名)
          "content":    str,   # 文本内容
          "time_stamp": str,   # 会话时间戳; LightMem 内部用 _parse_session_timestamp 解析,
                               #   仅接受 ISO 格式 (2023-05-08T13:56:00) 或
                               #   "2023/05/20 (Sat) 00:44"。其它格式必须在本适配器内先转成 ISO。
        }

    ── 数据集处理分派 (分发表) ──────────────────────────────────────────
    每个数据集一个转换函数, 在 ``_CONVERTERS`` 里按数据集名登记; ``_to_lightmem_message``
    只用名字选函数, 转换逻辑全在对应函数内, 名字未登记则走 ``_convert_default``。
    新增数据集 = 加一个 ``_convert_<name>`` 方法 + 在 ``_CONVERTERS`` 加一行 (格式兼容时
    什么都不用做, 自动走 default)。
      - locomo: event.timestamp 形如 "1:56 pm on 8 May, 2023" → 转成 ISO。
    """

    # LightMem stores LLM-rewritten summaries and retrieve() returns only
    # formatted strings, so retrieved memories cannot be reliably traced back to
    # source event ids from adapter-visible data. Evidence recall/mrr are N/A.
    supports_evidence_tracing = False

    def __init__(self) -> None:
        self._memory = None
        self._context: dict[str, Any] = {}
        self._events: dict[str, dict[str, MemoryEvent]] = {}
        self._config: dict[str, Any] = {}
        self._sample_id: str | None = None
        self._dataset: str = ""
        # Events are buffered during ingest and fed to LightMem in finalize_ingest,
        # so that add_memory is called with correctly-paired turns and force flags
        # (see _feed_pending_to_lightmem for the rationale).
        self._pending: list[MemoryEvent] = []

    @classmethod
    def preflight(cls) -> tuple[bool, str]:
        source_dir = Path(__file__).resolve().parents[2] / "systems" / "lightmem" / "source_selected"
        src_dir = source_dir / "src"
        if not src_dir.exists():
            return False, f"LightMem source missing: {src_dir}"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        if importlib.util.find_spec("lightmem.memory.lightmem") is None:
            return False, "LightMem module is not discoverable from the selected source"
        return True, "ready; use a Python environment with LightMem dependencies (imports deferred)"

    def initialize(self, context: dict[str, Any]) -> None:
        self._context = dict(context)
        self._events = {}
        self._dataset = str(context.get("dataset") or "").lower()
        self._config = self._load_config(context)
        source_dir = Path(
            self._config.get(
                "source_dir",
                Path(__file__).resolve().parents[2] / "systems" / "lightmem" / "source_selected",
            )
        ).resolve()
        src_dir = source_dir / "src"
        if not src_dir.exists():
            raise RuntimeError(f"LightMem source directory does not exist: {src_dir}")
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        try:
            module = importlib.import_module("lightmem.memory.lightmem")
            light_memory = getattr(module, "LightMemory")
        except Exception as exc:
            raise RuntimeError(
                "LightMem runtime cannot be imported (if its source is missing, run `python scripts/fetch_baselines.py` in eval/ first). Install the dependencies from "
                "systems/lightmem/source_selected/pyproject.toml in a dedicated environment. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        try:
            self._memory = light_memory.from_config(self._build_lightmem_config())
        except Exception as exc:
            raise RuntimeError(
                "LightMem initialization failed. Check LIGHTMEM_CONFIG, model paths, "
                "API credentials, and local Qdrant settings. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc

    def reset(self, user_id: str) -> None:
        self._events[user_id] = {}

    def begin_sample(self, sample_id: str) -> None:
        # Always (re)initialize per sample so every sample — including the first —
        # gets its own scoped Qdrant subdirectory (.qdrant/<sample_id>). Release
        # the previous sample's store first so Windows frees its .lock handle
        # before the next init.
        self._sample_id = sample_id
        self._pending = []
        self._release_memory()
        self.initialize(self._context)

    def _release_memory(self) -> None:
        """Best-effort release of the current LightMem/Qdrant handles.

        LightMemory exposes no close() method, but the per-sample scoped path is
        what actually prevents the WinError 32 lock race (a new sample never
        touches the previous sample's directory). Closing the underlying
        QdrantClient here is a bonus: it frees the .lock handle promptly so the
        old store can be cleaned up if desired.
        """
        if self._memory is None:
            return
        # Try LightMemory-level teardown if a future version adds one.
        for attr in ("close", "shutdown"):
            fn = getattr(self._memory, attr, None)
            if callable(fn):
                try:
                    fn()
                except Exception:  # noqa: BLE001 - best effort; never fail on cleanup
                    pass
        # Reach into the embedding retriever's QdrantClient and close it.
        retriever = getattr(self._memory, "embedding_retriever", None)
        client = getattr(retriever, "client", None)
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 - best effort
                pass
        self._memory = None

    def ingest(self, event: MemoryEvent) -> None:
        if self._memory is None:
            raise RuntimeError("LightMemAdapter is not initialized")
        self._events.setdefault(event.user_id, {})[event.event_id] = event
        # Defer the actual add_memory call: LightMem's segmenter needs correctly
        # paired turns and only the final turn should force segmentation/extraction.
        # Both are only knowable once the whole conversation has been seen.
        self._pending.append(event)

    def _to_lightmem_message(self, event: MemoryEvent) -> dict[str, Any]:
        """Convert a unified MemoryEvent into a LightMem message dict.

        Dispatch by dataset name to the registered converter; unregistered
        datasets fall back to ``_convert_default``. The name comes from the
        system config's ``_dataset`` field (via context["dataset"]).
        """
        converter = self._CONVERTERS.get(self._dataset, type(self)._convert_default)
        return converter(self, event)

    def _build_message(self, event: MemoryEvent, time_stamp: str) -> dict[str, Any]:
        """Common LightMem message shape shared by all dataset converters.

        Every source message becomes a ``role="user"`` message so LightMem's
        ``messages_use="user_only"`` extraction covers it; the original speaker
        (``event.role``) is kept in ``speaker_name`` — the extractor renders lines
        as ``"<speaker_name>: <content>"``, so identity survives without relying on
        ``event.role`` being a literal "user"/"assistant".
        """
        return {
            "role": "user",
            "content": event.content,
            "time_stamp": time_stamp,
            "speaker_name": event.role,
        }

    def _convert_default(self, event: MemoryEvent) -> dict[str, Any]:
        # Pass the timestamp through; LightMem accepts ISO or "YYYY/MM/DD (Ddd) HH:MM".
        return self._build_message(event, event.timestamp)

    def _convert_locomo(self, event: MemoryEvent) -> dict[str, Any]:
        # LoCoMo timestamps look like "1:56 pm on 8 May, 2023" -> normalize to ISO.
        return self._build_message(event, _normalize_locomo_timestamp(event.timestamp))

    # Dataset name -> converter. Keyed by the config's _dataset value (lowercased).
    # Add a new dataset = add a _convert_<name> method and one line here; formats
    # LightMem already accepts need no entry (they use _convert_default).
    _CONVERTERS = {
        "locomo": _convert_locomo,
    }

    def _feed_pending_to_lightmem(self) -> None:
        """Feed buffered events to LightMem as segmenter-compatible turns.

        LightMem's sensory buffer pairs messages two-by-two (``buffer[i]`` +
        ``buffer[i+1]``) as user/assistant turns, and ``user_only`` extraction
        keeps only ``role="user"`` lines. To satisfy both without dropping any
        source message, each real (user) message is paired with an empty
        ``assistant`` placeholder: the buffer stays even-length so pairing never
        indexes out of range, and every real message is still extracted.

        ``force_segment``/``force_extract`` are True only on the last turn — matching
        the official example (run_lightmem_transformers.py) — so intermediate turns
        accumulate and segment on token thresholds instead of one segment per message.
        """
        pending = self._pending
        self._pending = []
        for index, event in enumerate(pending):
            is_last = index == len(pending) - 1
            user_msg = self._to_lightmem_message(event)
            assistant_msg = {
                "role": "assistant",
                "content": "",
                "time_stamp": user_msg["time_stamp"],
                "speaker_name": "",
            }
            self._memory.add_memory(
                [user_msg, assistant_msg],
                force_segment=is_last,
                force_extract=is_last,
            )

    def finalize_ingest(self) -> None:
        if self._memory is None:
            return
        self._feed_pending_to_lightmem()
        if self._config.get("offline_update_all", False):
            # Official LOCOMO flow (add_locomo.py Phase 3): build update queue over
            # all accumulated entries, then offline-update with score_threshold=0.9.
            self._memory.construct_update_queue_all_entries()
            score_threshold = float(self._config.get("offline_update_score_threshold", 0.9))
            self._memory.offline_update_all_entries(score_threshold=score_threshold)

    def retrieve(
        self,
        query: str,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 10,
    ) -> list[MemoryRecord]:
        if self._memory is None:
            raise RuntimeError("LightMemAdapter is not initialized")
        raw_records = self._memory.retrieve(query, limit=top_k)
        records = []
        for index, content in enumerate(raw_records):
            text = str(content)
            source_ids = self._match_event_ids(user_id, text, session_id)
            records.append(
                MemoryRecord(
                    memory_id=f"lightmem-{user_id}-{index}",
                    content=text,
                    score=None,
                    source_event_ids=tuple(source_ids),
                    metadata={"system": "lightmem", "user_id": user_id},
                )
            )
        return records

    def _match_event_ids(self, user_id: str, text: str, session_id: str | None) -> list[str]:
        normalized = text.lower()
        matches = []
        for event_id, event in self._events.get(user_id, {}).items():
            if session_id and event.session_id != session_id:
                continue
            if event.content.lower() in normalized or normalized in event.content.lower():
                matches.append(event_id)
        return matches

    def dump_state(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "lightmem_adapter.json").write_text(
            json.dumps(
                {
                    "users": {user: list(events) for user, events in self._events.items()},
                    "config": self._config,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if self._memory is not None and hasattr(self._memory, "get_token_statistics"):
            (output_dir / "lightmem_token_statistics.json").write_text(
                json.dumps(self._memory.get_token_statistics(), indent=2), encoding="utf-8"
            )

    def close(self) -> None:
        self._release_memory()

    def _load_config(self, context: dict[str, Any]) -> dict[str, Any]:
        config_path = context.get("lightmem_config") or context.get("system_config") or os.getenv("LIGHTMEM_CONFIG")
        if not config_path:
            return {}
        path = Path(config_path)
        if not path.exists():
            raise RuntimeError(f"LightMem config does not exist: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _build_lightmem_config(self) -> dict[str, Any]:
        configured = dict(self._config.get("lightmem", self._config))
        spec = model_spec(self._config)
        if spec:
            manager = dict(configured.get("memory_manager", {}))
            manager_configs = dict(manager.get("configs", {}))
            manager["model_name"] = spec.get("provider", manager.get("model_name", "openai"))
            manager_configs["model"] = model_name(self._config, manager_configs.get("model", ""))
            if manager["model_name"] in {"openai", "openrouter", "deepseek"}:
                manager_configs["api_key"] = api_key(
                    self._config, _default_api_key_env(self._config), required=True
                )
                configured_base_url = base_url(self._config)
                if configured_base_url:
                    manager_configs["openai_base_url"] = configured_base_url
            manager["configs"] = manager_configs
            configured["memory_manager"] = manager
        text_embedder = dict(configured.get("text_embedder", {}))
        if text_embedder.get("model_name") == "openai":
            embedder_configs = dict(text_embedder.get("configs", {}))
            embedder_configs["api_key"] = api_key(
                self._config, _default_api_key_env(self._config), required=True
            )
            configured_base_url = base_url(self._config)
            if configured_base_url:
                embedder_configs["openai_base_url"] = configured_base_url
            text_embedder["configs"] = embedder_configs
            configured["text_embedder"] = text_embedder
        source_dir = configured.pop("source_dir", None)
        configured.pop("model", None)
        configured.pop("offline_update_all", None)
        if source_dir:
            self._config["source_dir"] = source_dir
        # Give each sample its own Qdrant subdirectory (``.qdrant/<sample_id>``)
        # instead of reusing one path. LightMem's Qdrant init does
        # ``shutil.rmtree(path)`` to wipe the store on every (re)initialize; on
        # Windows the previous sample's ``.lock`` handle may not be released yet,
        # so rmtree fails with ``WinError 32`` mid-run. Per-sample paths never
        # collide, so nothing needs deleting and the lock race disappears.
        self._scope_qdrant_path(configured)
        if configured:
            return configured
        return {
            "topic_segment": False,
            "metadata_generate": False,
            "text_summary": False,
            "index_strategy": "context",
            "retrieve_strategy": "context",
            "update": "offline",
        }

    def _scope_qdrant_path(self, configured: dict[str, Any]) -> None:
        """Point the Qdrant store at a per-sample subdirectory.

        Rewrites ``embedding_retriever.configs.path`` from ``<base>/.qdrant`` to
        ``<base>/.qdrant/<sample_id>`` so each LoCoMo sample gets an isolated
        store. Avoids the Windows ``WinError 32`` lock race caused by LightMem
        wiping (``shutil.rmtree``) a shared path that a previous sample still
        holds a handle to. No-op when the config has no qdrant path or no sample
        is active yet.
        """
        retriever = configured.get("embedding_retriever")
        if not isinstance(retriever, dict):
            return
        if str(retriever.get("model_name", "")).lower() != "qdrant":
            return
        configs = retriever.get("configs")
        if not isinstance(configs, dict):
            return
        base_path = configs.get("path")
        if not base_path or not self._sample_id:
            return
        # Sanitize the sample id so it is always a safe single path segment.
        safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(self._sample_id))
        configs["path"] = str(Path(base_path) / safe)


def _default_api_key_env(config: dict[str, Any]) -> str:
    provider = str(model_spec(config).get("provider", "openai")).lower()
    return {
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }.get(provider, "OPENAI_API_KEY")


def _normalize_locomo_timestamp(raw_ts: str) -> str:
    """LoCoMo timestamps look like "1:56 pm on 8 May, 2023".

    LightMem's session-time parser only accepts ISO or "YYYY/MM/DD (Ddd) HH:MM",
    so convert here to ISO. The known LoCoMo format is parsed with the stdlib
    (no external dependency); ``python-dateutil`` is used only as an optional
    fallback for unexpected variants. Raises ValueError if nothing can parse it,
    rather than passing a bad string through to LightMem's parser.
    """
    if not raw_ts:
        return raw_ts

    from datetime import datetime

    # Known LoCoMo format, e.g. "1:56 pm on 8 May, 2023" -> stdlib, no deps.
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %B %Y"):
        try:
            return datetime.strptime(raw_ts.strip(), fmt).isoformat()
        except ValueError:
            pass

    # Optional lenient fallback for unexpected variants.
    try:
        from dateutil import parser as _dtparser

        return _dtparser.parse(raw_ts.replace(" on ", " ")).isoformat()
    except ImportError:
        pass
    except Exception:
        pass

    raise ValueError(
        f"Cannot normalize LoCoMo timestamp {raw_ts!r} to ISO. "
        f"Expected e.g. '1:56 pm on 8 May, 2023'. "
        f"Install 'python-dateutil' for lenient parsing of other variants."
    )
