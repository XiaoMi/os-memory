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
from typing import Any

from hypereval.protocol import BenchmarkSample


class DatasetAdapter(ABC):
    name: str

    @abstractmethod
    def load(self, path: str | None = None, sample_size: int | None = None) -> list[BenchmarkSample]:
        raise NotImplementedError

    def metadata(self) -> dict[str, Any]:
        return {"name": self.name}
