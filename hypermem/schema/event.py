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

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from hypermem.schema.serialization import (
    JsonModel, parse_datetime, require_anchors, require_embedding,
    require_number, require_strings, require_text, utc_datetime,
)


@dataclass(frozen=True)
class CanonicalEvent(JsonModel):
    event_id: str
    start_time: datetime
    end_time: datetime
    state: str
    entities: list[str]
    anchors: dict[str, list[str]]
    observation_ids: list[str]
    confidence: float
    embedding: list[float] | None = None

    def __post_init__(self) -> None:
        require_text(self.event_id, "event_id")
        object.__setattr__(self, "start_time", utc_datetime(self.start_time))
        object.__setattr__(self, "end_time", utc_datetime(self.end_time))
        if self.end_time < self.start_time:
            raise ValueError("event end_time precedes start_time")
        require_text(self.state, "canonical state")
        require_strings(self.entities, "entities")
        require_anchors(self.anchors)
        require_strings(self.observation_ids, "observation_ids")
        if not self.observation_ids:
            raise ValueError("events must have at least one supporting observation")
        require_number(self.confidence, "confidence", unit_interval=True)
        require_embedding(self.embedding)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CanonicalEvent:
        return cls(**{
            **value,
            "start_time": parse_datetime(value["start_time"]),
            "end_time": parse_datetime(value["end_time"]),
        })


RELATION_TYPES = frozenset({"before", "after", "same_entity", "same_anchor", "state_transition", "same_episode"})


@dataclass(frozen=True)
class EventRelation(JsonModel):
    source_event: str
    target_event: str
    relation_type: str
    score: float

    def __post_init__(self) -> None:
        require_text(self.source_event, "source_event")
        require_text(self.target_event, "target_event")
        if self.source_event == self.target_event:
            raise ValueError("event relations must connect distinct events")
        if self.relation_type not in RELATION_TYPES:
            raise ValueError(f"unsupported relation_type: {self.relation_type}")
        require_number(self.score, "relation score", unit_interval=True)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EventRelation:
        return cls(**value)
