# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""Lossless PMM MemoryEvent envelope; gold QA never enters this adapter."""
from dataclasses import replace
from datetime import datetime
import hashlib
import json

from hypermem.adapters.os_trace import OSTraceAdapter


def identity(source_id, line):
    # Lexicographic order preserves source order for equal timestamps.
    return hashlib.sha256(source_id.encode()).hexdigest() + f"-{line:012d}"


def unpack_record(record):
    if record.get("role") != "user" or not isinstance(record.get("content"), str):
        raise ValueError("Expected a PMM raw MemoryEvent")
    raw = json.loads(record["content"])
    meta = record["metadata"]
    line = meta["input_line"]
    if (isinstance(line, bool) or not isinstance(line, int) or line < 1
            or record["event_id"] != f"raw-line:{line}"
            or meta["source_lines"] != [line] or meta["input_kind"] != "raw"):
        raise ValueError("Invalid PMM source-line mapping")
    if not isinstance(record.get("user_id"), str) or not record["user_id"]:
        raise ValueError("Missing PMM user namespace")
    return raw


class PMMAdapter:
    def parse(self, record, *, observation_id, provenance):
        raw = unpack_record(record)
        observation = OSTraceAdapter().parse(raw, observation_id=observation_id, provenance=provenance)
        timestamp = datetime.fromisoformat(record["timestamp"])
        if timestamp.tzinfo is None or timestamp != observation.start_time:
            raise ValueError("PMM envelope timestamp differs from raw timestamp")
        if provenance.metadata["user_id"] != record["user_id"]:
            raise ValueError("Mixed PMM users")
        # Not just payload.event: the entire, byte-exact original JSON text.
        return replace(observation, content=record["content"])
