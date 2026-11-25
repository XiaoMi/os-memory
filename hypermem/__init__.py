# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""HyperOS-Mem V11 core data layer. Importing requires no model services.

The data layer above is imported eagerly. The public library interface,
``HyperMemory`` (with ``format_memories`` and ``ModelServiceError``), is loaded
lazily on first access so that ``import hypermem`` stays free of the
engine/SAGE modules and of any model service. See ``hypermem.api`` and README.md at the repository root.
"""

__version__ = "1.0.0"

from hypermem.memory import EventMemory
from hypermem.schema import (
    AtomicObservation,
    CanonicalEvent,
    EventRelation,
    ObservationFeatures,
    Provenance,
    TimeRange,
)

__all__ = [
    "AtomicObservation", "CanonicalEvent", "EventMemory", "EventRelation",
    "HyperMemory", "ModelServiceError", "ObservationFeatures", "Provenance", "TimeRange",
    "format_memories", "__version__",
]

_LAZY = {"HyperMemory", "ModelServiceError", "format_memories"}


def __getattr__(name):
    # PEP 562: load the library interface only when actually requested, so the
    # bare data layer stays importable without the online/SAGE dependencies.
    if name in _LAZY:
        from hypermem import api
        return getattr(api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
