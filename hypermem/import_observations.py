# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""Offline JSONL validation/import for milestone 1–2. Does not form events."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any

from hypermem.adapters import OSTraceAdapter
from hypermem.memory import EventMemory
from hypermem.schema import Provenance
from hypermem.schema.serialization import encode_json, require_text


def _finite_json(value: str) -> None:
    raise ValueError("non-finite JSON number")


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def import_jsonl(
    path: str | Path,
    *,
    source_id: str,
    memory: EventMemory | None = None,
    limit: int | None = None,
    include_synthetic_location: bool = False,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    input_format: str = "os_trace",
    user_id: str | None = None,
) -> dict[str, Any]:
    """Validate all selected observations and optionally persist atomically.

    The caller assigns a stable dataset namespace (source_id). IDs depend on
    namespace + physical line number, never on event identity or model output.
    A failed import rolls back all writes made by this call. No source data is
    modified, no media is opened, and no network service is used.
    """
    require_text(source_id, "source_id")
    if input_format not in {"os_trace", "atomic_bundle", "pmm", "pmm_cleaned"}:
        raise ValueError("unsupported input format")
    if user_id is not None:
        require_text(user_id, "user_id")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise ValueError("limit must be a positive integer or None")
    for value in (start_time,end_time):
        if value is not None and (not isinstance(value,datetime) or value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("Import window bounds must be timezone-aware datetimes")
    if start_time is not None and end_time is not None and start_time >= end_time:
        raise ValueError("Import window must have start_time < end_time")
    path = Path(path).expanduser().resolve()
    types: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    accepted = inserted = duplicates = lines = blank_lines = backwards = outside_window = 0
    earliest = latest = previous = None
    scanned_hash = hashlib.sha256()
    adapter = OSTraceAdapter()
    scan_complete = True
    allowed = {"real"}
    seen_upstream_ids = set()
    if input_format == "atomic_bundle":
        from hypermem.adapters.atomic_bundle import AtomicBundleAdapter, BUNDLE_TYPES
        adapter = AtomicBundleAdapter()
        allowed = BUNDLE_TYPES
    if input_format == "pmm":
        from hypermem.adapters.pmm import PMMAdapter
        adapter = PMMAdapter()
        allowed = {"real", "wifi_segment", "gps_segment"}
    if input_format == "pmm_cleaned":
        allowed = {"real"}
    if include_synthetic_location:
        allowed.update({"wifi_segment", "gps_segment"})
    with path.open("rb") as stream, memory.transaction() if memory else nullcontext():
        while True:
            offset = stream.tell()
            raw_line = stream.readline()
            if not raw_line:
                break
            lines += 1
            scanned_hash.update(raw_line)
            try:
                decoded = raw_line.decode("utf-8-sig" if lines == 1 else "utf-8")
                if not decoded.strip():
                    blank_lines += 1
                    continue
                raw = json.loads(decoded, parse_constant=_finite_json, object_pairs_hook=_unique_keys)
                if not isinstance(raw, dict):
                    raise ValueError("JSONL record must be an object")
                inner = raw
                if input_format == "pmm":
                    from hypermem.adapters.pmm import unpack_record, identity as pmm_identity
                    inner = unpack_record(raw)
                if input_format == "pmm_cleaned":
                    cleaning = raw.get("_cleaning")
                    if not isinstance(cleaning, dict) or not isinstance(cleaning.get("source_lines"), list) or not cleaning["source_lines"]:
                        raise ValueError("PMM cleaned record lacks original source lines")
                    if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in cleaning["source_lines"]):
                        raise ValueError("Invalid PMM cleaned source lines")
                kind = inner.get("observation_type" if input_format == "atomic_bundle" else "type")
                require_text(kind, "record type")
                types[kind] += 1
                if kind not in allowed:
                    if input_format in {"atomic_bundle", "pmm", "pmm_cleaned"}:
                        raise ValueError("unsupported bundle type; refusing silent exclusion")
                    skipped[kind] += 1
                    continue
                identity = hashlib.sha256(encode_json([source_id, lines]).encode("utf-8")).hexdigest()
                if input_format in {"pmm", "pmm_cleaned"}:
                    from hypermem.adapters.pmm import identity as pmm_identity
                    identity = pmm_identity(source_id, lines)
                payload = inner.get("payload")
                metadata = {
                    "source_id": source_id,
                    "line_number": lines,
                    "byte_offset": offset,
                    "raw_line_sha256": hashlib.sha256(raw_line).hexdigest(),
                    "upstream_src_line": payload.get("src_line") if isinstance(payload, dict) else None,
                    "synthetic": kind != "real",
                }
                if user_id is not None:
                    metadata["user_id"] = user_id
                if input_format == "pmm":
                    if user_id is not None and user_id != raw["user_id"]:
                        raise ValueError("Unexpected PMM user")
                    metadata.update(input_format="pmm", user_id=raw["user_id"],
                                    upstream_event_id=raw["event_id"],
                                    source_lines=raw["metadata"]["source_lines"],
                                    input_line=raw["metadata"]["input_line"])
                if input_format == "pmm_cleaned":
                    metadata.update(input_format="pmm_cleaned", source_lines=sorted(set(raw["_cleaning"]["source_lines"])),
                                    upstream_observation_id=raw["_cleaning"].get("observation_id"),
                                    repeat_count=raw["_cleaning"].get("repeat_count"))
                if input_format == "atomic_bundle":
                    upstream_id = raw.get("observation_id")
                    require_text(upstream_id, "upstream observation_id")
                    if upstream_id in seen_upstream_ids:
                        raise ValueError("duplicate upstream observation ID")
                    seen_upstream_ids.add(upstream_id)
                    metadata.update(input_format=input_format, synthetic=False,
                                    upstream_observation_id=upstream_id,
                                    aggregation_level=raw.get("aggregation_level"),
                                    upstream_evidence=raw.get("evidence"),
                                    upstream_provenance=raw.get("provenance"))
                observation = adapter.parse(
                    raw,
                    observation_id=f"O-{identity}",
                    provenance=Provenance(f"R-{identity}", str(path), kind, metadata),
                )
                # Both directions are validated even in --validate-only mode.
                type(observation).from_json(observation.to_json())
                if (start_time is not None and observation.start_time < start_time or
                        end_time is not None and observation.start_time >= end_time):
                    outside_window += 1
                    continue
                if memory is not None:
                    if memory.add_observation(observation, raw):
                        inserted += 1
                    else:
                        duplicates += 1
                accepted += 1
                backwards += int(previous is not None and observation.start_time < previous)
                if input_format in {"pmm", "pmm_cleaned"} and backwards:
                    raise ValueError("PMM records must already be chronologically sorted")
                previous = observation.start_time
                earliest = min(earliest, observation.start_time) if earliest else observation.start_time
                latest = max(latest, observation.end_time) if latest else observation.end_time
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                # Do not include raw personal content in diagnostic exceptions.
                raise ValueError(f"Invalid record at JSONL line {lines}: {type(exc).__name__}") from exc
            if limit is not None and accepted >= limit:
                scan_complete = stream.peek(1) == b""
                break
    report = {
        "schema_version": 1,
        "operation": "validate_observations" if memory is None else "import_observations",
        "config": {
            "input": str(path), "source_id": source_id, "limit": limit,
            "include_synthetic_location": include_synthetic_location,
            "timestamp_unit": "unix_milliseconds", "storage_timezone": "UTC",
            "input_format": input_format, "user_id": user_id,
        },
        "scanned_bytes_sha256": scanned_hash.hexdigest(),
        "scan_complete": scan_complete,
        "lines_scanned": lines,
        "blank_lines": blank_lines,
        "record_types": dict(sorted(types.items())),
        "skipped_types": dict(sorted(skipped.items())),
        "accepted_observations": accepted,
        "inserted_observations": inserted,
        "duplicate_observations": duplicates,
        "out_of_order_observations": backwards,
        "start_time": earliest.isoformat() if earliest else None,
        "end_time": latest.isoformat() if latest else None,
        "event_formation_performed": False,
        "model_calls": 0,
    }
    if start_time is not None or end_time is not None:
        report["config"]["time_window"] = {"start_inclusive": start_time.isoformat() if start_time else None,
                                         "end_exclusive": end_time.isoformat() if end_time else None,
                                         "field": "observation.start_time"}
        report["outside_time_window"] = outside_window
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-id", required=True, help="Stable dataset namespace for observation/raw IDs")
    parser.add_argument("--output", type=Path, help="New run directory; existing directories are never overwritten")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--limit", type=int, help="Stop after this many accepted observations")
    parser.add_argument("--include-synthetic-location", action="store_true")
    parser.add_argument("--input-format", choices=("os_trace", "atomic_bundle", "pmm", "pmm_cleaned"), default="os_trace")
    parser.add_argument("--user-id", help="Explicit owner namespace, not inferred from message contents")
    args = parser.parse_args(argv)
    if not args.validate_only and args.output is None:
        parser.error("--output is required for a persistent import")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    try:
        if args.output is not None:
            args.output.mkdir(parents=True, exist_ok=False)
        database = args.output / "memory.sqlite" if not args.validate_only else None
        with EventMemory(database) if database is not None else nullcontext() as memory:
            report = import_jsonl(
                args.input, source_id=args.source_id, memory=memory, limit=args.limit,
                include_synthetic_location=args.include_synthetic_location,
                input_format=args.input_format, user_id=args.user_id,
            )
            if memory is not None:
                report["storage_counts"] = memory.counts()
                memory.set_metadata("import_report", report)
        result = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output is not None:
            (args.output / "import_report.json").write_text(result + "\n", encoding="utf-8")
        print(result)
        return 0
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(f"Import failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
