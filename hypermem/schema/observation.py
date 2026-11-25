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

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from hypermem.schema.serialization import (
    JsonModel, json_value, parse_datetime, require_anchors, require_embedding,
    require_strings, require_text, utc_datetime,
)


@dataclass(frozen=True)
class TimeRange(JsonModel):
    """An interval in the paper's time field; equal endpoints are valid."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", utc_datetime(self.start))
        object.__setattr__(self, "end", utc_datetime(self.end))
        if self.end < self.start:
            raise ValueError("time range end precedes start")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TimeRange:
        return cls(**{**value, "start": parse_datetime(value["start"]), "end": parse_datetime(value["end"])})


@dataclass(frozen=True)
class Provenance(JsonModel):
    raw_id: str
    raw_path: str | None = None
    source_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_text(self.raw_id, "raw_id")
        for name in ("raw_path", "source_type"):
            if getattr(self, name) is not None:
                require_text(getattr(self, name), name)
        if not isinstance(self.metadata, dict):
            raise ValueError("provenance metadata must be a JSON object")
        json_value(self.metadata)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Provenance:
        return cls(**value)


@dataclass(frozen=True)
class AtomicObservation(JsonModel):
    """Five intrinsic fields plus an engineering ID; no embeddings/entities.

    timestamp is either a point or an interval. Derived features are stored
    separately, keyed by observation_id, and never inferred by this schema.
    """

    observation_id: str
    timestamp: datetime | TimeRange
    source: str
    content: str
    anchors: dict[str, list[str]]
    provenance: Provenance

    def __post_init__(self) -> None:
        require_text(self.observation_id, "observation_id")
        if not isinstance(self.timestamp, TimeRange):
            object.__setattr__(self, "timestamp", utc_datetime(self.timestamp))
        require_text(self.source, "source")
        if not isinstance(self.content, str):
            raise ValueError("content must be a string")
        require_anchors(self.anchors)
        if not isinstance(self.provenance, Provenance):
            raise ValueError("provenance must be a Provenance object")

    @property
    def start_time(self) -> datetime:
        return self.timestamp.start if isinstance(self.timestamp, TimeRange) else self.timestamp

    @property
    def end_time(self) -> datetime:
        return self.timestamp.end if isinstance(self.timestamp, TimeRange) else self.timestamp

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AtomicObservation:
        timestamp = value["timestamp"]
        return cls(**{
            **value,
            "timestamp": TimeRange.from_dict(timestamp) if isinstance(timestamp, dict) else parse_datetime(timestamp),
            "provenance": Provenance.from_dict(value["provenance"]),
        })


@dataclass(frozen=True)
class ObservationFeatures(JsonModel):
    """Replaceable derived features; metadata can record model name/revision."""

    embedding: list[float] | None = None
    entities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_embedding(self.embedding)
        require_strings(self.entities, "entities")
        if not isinstance(self.metadata, dict):
            raise ValueError("feature metadata must be a JSON object")
        json_value(self.metadata)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ObservationFeatures:
        return cls(**value)
