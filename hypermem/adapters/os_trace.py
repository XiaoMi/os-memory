# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""Mechanical conversion of the supplied OS JSONL envelope, without event inference."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from hypermem.schema import AtomicObservation, Provenance, TimeRange
from hypermem.schema.serialization import encode_json


_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_ANCHOR_KEYS = ("order_id", "transaction_id", "tracking_id", "reservation_id", "phone", "email", "url")


def milliseconds(value: Any) -> datetime:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("OS ts and ts_end must be integer Unix milliseconds")
    return _EPOCH + timedelta(milliseconds=value)


class OSTraceAdapter:
    """Supports real OS records and explicitly opted-in synthetic location records.

    Segment summaries are not Atomic Observations. The complete event object is
    serialized as content so no fields (including existing captions) are lost.
    Only explicitly named anchor fields are copied; no entities, event labels,
    regex inference, or model-derived features are created here.
    """

    def parse(self, raw_record: dict[str, Any], *, observation_id: str, provenance: Provenance) -> AtomicObservation:
        if raw_record.get("type") not in {"real", "wifi_segment", "gps_segment"}:
            raise ValueError("unsupported atomic record type (segment summaries are not atomic)")
        payload = raw_record.get("payload")
        if not isinstance(payload, dict) or not isinstance(payload.get("event"), dict):
            raise ValueError("payload.event must be an object")
        event = payload["event"]
        source = event.get("from_pkg") or event.get("from_app") or event.get("perception_type")
        start = milliseconds(raw_record["ts"])
        end = milliseconds(raw_record.get("ts_end", raw_record["ts"]))
        interval = TimeRange(start, end)
        anchors: dict[str, list[str]] = {}
        data = event.get("data", {})
        if not isinstance(data, dict):
            raise ValueError("payload.event.data must be an object")
        for key in _ANCHOR_KEYS:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                anchors[key] = [value]
        return AtomicObservation(
            observation_id=observation_id,
            timestamp=start if start == end else interval,
            source=source,
            content=encode_json(event),
            anchors=anchors,
            provenance=provenance,
        )
