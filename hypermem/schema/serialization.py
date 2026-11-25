# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""Strict JSON serialization shared by the data models (no provider dependencies)."""
from __future__ import annotations

import json
import math
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from typing import Any


def json_value(value: Any) -> Any:
    # Vectors account for most leaves in formation snapshots. Exact builtin
    # types cannot be dataclasses/containers; keep subclass handling below.
    kind = type(value)
    if kind is float:
        if not math.isfinite(value):
            raise ValueError('Not a finite JSON value: float')
        return value
    if value is None or kind in (str, bool, int):
        return value
    if isinstance(value, datetime):
        return utc_datetime(value).isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"Not a finite JSON value: {type(value).__name__}")


def encode_json(value: Any) -> str:
    return json.dumps(json_value(value), ensure_ascii=False, sort_keys=True, allow_nan=False)


def require_text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def require_strings(value: Any, name: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of strings")
    for item in value:
        require_text(item, name)
    if len(value) != len(set(value)):
        raise ValueError(f"{name} must not contain duplicates")


def require_anchors(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("anchors must be a mapping of types to string lists")
    for key, items in value.items():
        require_text(key, "anchor type")
        require_strings(items, "anchor values")


def require_number(value: Any, name: str, *, unit_interval: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if unit_interval and not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")


def require_embedding(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, list) or not value:
        raise ValueError("embedding must be None or a nonempty numeric list")
    for item in value:
        # Preserve the scalar validator for integers, booleans and subclasses.
        # Normal model vectors avoid one Python call per dimension per read.
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError('embedding value must be a finite number')
        else:
            require_number(item, "embedding value")


def utc_datetime(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware datetimes")
    return value.astimezone(timezone.utc)


def parse_datetime(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("serialized timestamps must be ISO 8601 strings")
    return utc_datetime(datetime.fromisoformat(value.replace("Z", "+00:00")))


class JsonModel:
    def to_dict(self) -> dict[str, Any]:
        return json_value(self)

    def to_json(self) -> str:
        return encode_json(self)

    @classmethod
    def from_json(cls, value: str):
        return cls.from_dict(json.loads(value))
