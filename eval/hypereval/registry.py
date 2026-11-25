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
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hypereval.adapters import (
    AMEMAdapter,
    LightMemAdapter,
    LightRAGAdapter,
    Mem0Adapter,
    MockMemoryAdapter,
    PlugMemAdapter,
    SimpleMemAdapter,
)
from hypereval.datasets import (
    GalleryMemDataset,
    LoCoMoDataset,
    LongMemEvalDataset,
    MemBenchDataset,
    OsEventsDataset,
    PersonaMemDataset,
    ToyDataset,
)


@dataclass(frozen=True)
class RegisteredSystem:
    name: str
    display_name: str
    status: str
    repository: str | None
    source_dir: str | None
    adapter: str | None
    manifest_path: Path | None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _manifest(name: str) -> tuple[dict[str, Any], Path | None]:
    path = _project_root() / "systems" / name / "manifest.json"
    if not path.exists():
        return {}, None
    return json.loads(path.read_text(encoding="utf-8")), path


def list_systems() -> list[RegisteredSystem]:
    result = []
    systems_dir = _project_root() / "systems"
    if systems_dir.exists():
        for path in sorted(systems_dir.iterdir()):
            if not path.is_dir():
                continue
            data, manifest_path = _manifest(path.name)
            result.append(RegisteredSystem(
                name=data.get("name", path.name),
                display_name=data.get("display_name", path.name),
                status=data.get("status", "source-only"),
                repository=data.get("repository"),
                source_dir=data.get("source_dir"),
                adapter=data.get("adapter"),
                manifest_path=manifest_path,
            ))
    return result


def create_adapter(name: str):
    if name == "mock":
        return MockMemoryAdapter()
    if name == "lightmem":
        return LightMemAdapter()
    if name == "amem":
        return AMEMAdapter()
    if name == "mem0":
        return Mem0Adapter()
    if name == "simplemem":
        return SimpleMemAdapter()
    if name == "plugmem":
        return PlugMemAdapter()
    if name == "lightrag":
        return LightRAGAdapter()
    data, _ = _manifest(name)
    adapter_path = data.get("adapter")
    if not adapter_path:
        raise RuntimeError(f"System {name!r} has no adapter yet; its status is source-only")
    module_name, class_name = adapter_path.split(":", 1)
    return getattr(importlib.import_module(module_name), class_name)()


def create_dataset(name: str):
    if name == "toy":
        return ToyDataset()
    if name == "locomo":
        return LoCoMoDataset()
    if name == "longmemeval":
        return LongMemEvalDataset()
    if name == "gallerymem":
        return GalleryMemDataset()
    if name == "membench":
        return MemBenchDataset()
    if name == "os_events":
        return OsEventsDataset()
    if name == "personamem":
        return PersonaMemDataset()
    raise KeyError(f"Unknown dataset: {name}")


def check_system(name: str) -> tuple[bool, str]:
    if name == "mock":
        return True, "ready"
    data, path = _manifest(name)
    if path is None:
        return False, "manifest missing"
    source_dir = data.get("source_dir")
    if source_dir:
        source_path = _project_root() / source_dir
        if not source_path.exists() and data.get("status") == "ready":
            return False, f"source missing: {source_path}"
    if data.get("status") != "ready":
        return True, data.get("status", "source-only")
    if name == "lightmem":
        ready, message = LightMemAdapter.preflight()
    elif name == "amem":
        ready, message = AMEMAdapter.preflight()
    elif name == "mem0":
        ready, message = Mem0Adapter.preflight()
    elif name == "simplemem":
        ready, message = SimpleMemAdapter.preflight()
    elif name == "plugmem":
        ready, message = PlugMemAdapter.preflight()
    elif name == "lightrag":
        ready, message = LightRAGAdapter.preflight()
    else:
        try:
            create_adapter(name)
        except Exception as exc:
            return False, str(exc)
        return True, "ready"
    runtime_python = data.get("runtime_python")
    if ready and runtime_python and not Path(runtime_python).exists():
        return False, f"declared runtime missing: {runtime_python}"
    return ready, message
