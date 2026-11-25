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

from abc import ABC, abstractmethod
from pathlib import Path
import shutil
from typing import Any

from hypereval.protocol import MemoryEvent, MemoryRecord


def clear_sample_storage(path: str | Path, run_dir: str | Path) -> None:
    target = Path(path).resolve()
    root = Path(run_dir).resolve()
    if target == root or not target.is_relative_to(root):
        raise RuntimeError(
            f"Refusing to clear sample storage outside run directory: {target}"
        )
    if target.exists():
        shutil.rmtree(target)


class MemoryAdapter(ABC):
    """Smallest contract required by the benchmark runner."""

    # Whether retrieved records can be reliably traced back to the source event
    # ids (dataset-native evidence ids). Extraction/rewriting systems whose
    # memories are LLM-synthesized summaries cannot do this from adapter-visible
    # data, so they override this to False and the runner records evidence
    # recall / mrr as N/A (None) instead of a misleading 0.
    supports_evidence_tracing: bool = True

    # True only when repeated ingest/finalize calls on one isolated store are
    # the system's native online-write behavior. PersonaMem uses this capability
    # to add only events introduced at each later checkpoint. Batch structuring
    # systems keep the default False because splitting a trajectory can change
    # the memories they produce.
    supports_incremental_checkpoints: bool = False

    # MemBench Capacity is a live message stream with a read after every turn
    # following the evidence. Systems that can faithfully continue writing after
    # those reads opt in here. Trajectory-finalization systems keep False and the
    # runner rebuilds an exact prefix instead of inventing an online mode.
    supports_capacity_streaming: bool = False

    @abstractmethod
    def initialize(self, context: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset(self, user_id: str) -> None:
        raise NotImplementedError

    def begin_sample(self, sample_id: str) -> None:
        """Start an isolated benchmark sample.

        Stateful systems can clear or recreate their store here. Stateless
        adapters may keep the default no-op implementation.
        """
        return None

    @abstractmethod
    def ingest(self, event: MemoryEvent) -> None:
        raise NotImplementedError

    def finalize_ingest(self) -> None:
        return None

    def ingest_capacity_event(self, event: MemoryEvent) -> None:
        """Process one event through the system's native online-write path."""
        self.ingest(event)
        self.finalize_ingest()

    def finalize_capacity_sample(self) -> None:
        """Flush remaining online buffers after the last scored checkpoint."""
        return None

    @abstractmethod
    def retrieve(
        self,
        query: str,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 10,
    ) -> list[MemoryRecord]:
        raise NotImplementedError

    def dump_state(self, output_dir: Path) -> None:
        return None

    def close(self) -> None:
        return None
