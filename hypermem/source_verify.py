# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""Exact full-source/raw-record/provenance correspondence check; no import or model calls."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from hypermem.memory.readonly import now, readonly
from hypermem.memory import EventMemory
from hypermem.schema.serialization import encode_json


def sha(value):
    return hashlib.sha256(value).hexdigest()


def verify_source(run, *, allow_prefix=False, allow_window=False):
    run = Path(run).resolve()
    report = json.loads((run / "import_report.json").read_text(encoding="utf-8"))
    source = Path(report["config"]["input"]).resolve()
    namespace = report["config"]["source_id"]
    limit = report["config"]["limit"] if allow_prefix else None
    window = report["config"].get("time_window")
    start = datetime.fromisoformat(window["start_inclusive"]) if window and window["start_inclusive"] else None
    end = datetime.fromisoformat(window["end_exclusive"]) if window and window["end_exclusive"] else None
    allowed = {"real"}
    bundle = report["config"].get("input_format") == "atomic_bundle"
    pmm = report["config"].get("input_format") == "pmm"
    pmm_cleaned = report["config"].get("input_format") == "pmm_cleaned"
    if pmm:
        allowed = {"real", "wifi_segment", "gps_segment"}
    if pmm_cleaned:
        allowed = {"real"}
    if bundle:
        from hypermem.adapters.atomic_bundle import BUNDLE_TYPES
        allowed = BUNDLE_TYPES
    if report["config"]["include_synthetic_location"]:
        allowed.update({"wifi_segment", "gps_segment"})
    expected, types = {}, Counter()
    total_hash, lines, blank, offset = hashlib.sha256(), 0, 0, 0
    with source.open("rb") as stream:
        for raw_line in stream:
            lines += 1
            total_hash.update(raw_line)
            line_offset, offset = offset, offset + len(raw_line)
            if not raw_line.strip():
                blank += 1
                continue
            raw = json.loads(raw_line.decode("utf-8-sig" if lines == 1 else "utf-8"))
            inner = json.loads(raw["content"]) if pmm else raw
            kind = inner["observation_type" if bundle else "type"]
            types[kind] += 1
            if kind not in allowed:
                continue
            if window:
                from hypermem.adapters.os_trace import milliseconds
                timestamp = milliseconds(raw["timestamp_ms" if bundle else "ts"])
                if start is not None and timestamp < start or end is not None and timestamp >= end:
                    continue
            identity = sha(encode_json([namespace, lines]).encode())
            if pmm or pmm_cleaned:
                from hypermem.adapters.pmm import identity as pmm_identity
                identity = pmm_identity(namespace, lines)
            expected[identity] = (sha(encode_json(raw).encode()), sha(raw_line), lines, line_offset, kind)
            if limit is not None and len(expected) >= limit:
                break
        actual_scan_complete = stream.read(1) == b""
    problems, problem_count = [], 0
    def issue(code, detail):
        nonlocal problem_count
        problem_count += 1
        if len(problems) < 100:
            problems.append({"code": code, "detail": detail})
    if total_hash.hexdigest() != report["scanned_bytes_sha256"] or actual_scan_complete != report["scan_complete"]:
        issue("source_fingerprint_or_scan_mismatch", "Current scanned bytes differ from the recorded import")
    if not allow_prefix and (not report["scan_complete"] or report["config"]["limit"] is not None):
        issue("full_scan_required", "Prefix verification requires explicit allow_prefix; this is not a full-source check")
    if window and not allow_window:
        issue("window_requires_opt_in", "Time-window verification requires explicit allow_window; this is not all selected-type records")
    if len(expected) != report["accepted_observations"] or lines != report["lines_scanned"] or dict(types) != report["record_types"]:
        issue("import_counts_mismatch", "Input record counts differ from import report")
    remaining_raw, remaining_observations = set(expected), set(expected)
    checked_raw = checked_observations = iterated_observations = 0
    remaining_iteration = set(expected)
    with readonly(run) as conn:
        for row in conn.execute("SELECT raw_id,raw_json FROM raw_records"):
            checked_raw += 1
            identity = row[0][2:] if row[0].startswith("R-") else None
            if identity not in expected:
                issue("unexpected_raw_record", row[0])
                continue
            if sha(row[1].encode()) != expected[identity][0]:
                issue("raw_record_content_mismatch", row[0])
            remaining_raw.discard(identity)
        for row in conn.execute("SELECT observation_id,raw_id,json_extract(observation_json,'$.provenance') FROM observations"):
            checked_observations += 1
            identity = row[0][2:] if row[0].startswith("O-") else None
            if identity not in expected:
                issue("unexpected_observation", row[0])
                continue
            _, line_hash, line_number, byte_offset, source_type = expected[identity]
            provenance = json.loads(row[2])
            meta = provenance["metadata"]
            actual = (row[1], provenance["raw_id"], provenance["raw_path"], provenance["source_type"],
                      meta["source_id"], meta["raw_line_sha256"], meta["line_number"], meta["byte_offset"])
            wanted = ("R-" + identity, "R-" + identity, str(source), source_type,
                      namespace, line_hash, line_number, byte_offset)
            if actual != wanted:
                issue("observation_provenance_mismatch", row[0])
            remaining_observations.discard(identity)
        previous = None
        # Exercise the production reader against the entire actual input store,
        # through a read-only connection and without constructing a writer.
        for observation in EventMemory.iter_observations(SimpleNamespace(_connection=conn)):
            iterated_observations += 1
            identity = observation.observation_id[2:] if observation.observation_id.startswith("O-") else None
            if identity not in remaining_iteration:
                issue("unexpected_or_duplicate_iterator_record", observation.observation_id)
            remaining_iteration.discard(identity)
            if bundle:
                raw = json.loads(conn.execute("SELECT raw_json FROM raw_records WHERE raw_id=?", (observation.provenance.raw_id,)).fetchone()[0])
                from hypermem.adapters.atomic_bundle import AtomicBundleAdapter
                reconstructed = AtomicBundleAdapter().parse(raw, observation_id=observation.observation_id, provenance=observation.provenance)
                if reconstructed != observation or observation.provenance.metadata.get("upstream_observation_id") != raw["observation_id"]:
                    issue("bundle_observation_content_mismatch", observation.observation_id)
            if pmm:
                raw = json.loads(conn.execute("SELECT raw_json FROM raw_records WHERE raw_id=?", (observation.provenance.raw_id,)).fetchone()[0])
                from hypermem.adapters.pmm import PMMAdapter
                reconstructed = PMMAdapter().parse(raw, observation_id=observation.observation_id, provenance=observation.provenance)
                if reconstructed != observation or observation.provenance.metadata.get("source_lines") != raw["metadata"]["source_lines"]:
                    issue("pmm_observation_content_mismatch", observation.observation_id)
            if pmm_cleaned:
                raw = json.loads(conn.execute("SELECT raw_json FROM raw_records WHERE raw_id=?", (observation.provenance.raw_id,)).fetchone()[0])
                from hypermem.adapters.os_trace import OSTraceAdapter
                reconstructed = OSTraceAdapter().parse(raw, observation_id=observation.observation_id, provenance=observation.provenance)
                if reconstructed != observation or observation.provenance.metadata.get("source_lines") != sorted(set(raw["_cleaning"]["source_lines"])):
                    issue("pmm_cleaned_source_mapping_mismatch", observation.observation_id)
            position = (observation.start_time.isoformat(), observation.observation_id)
            if previous is not None and position <= previous:
                issue("iterator_order_mismatch", observation.observation_id)
            previous = position
    if remaining_raw or remaining_observations:
        issue("missing_input_records", {"raw": len(remaining_raw), "observations": len(remaining_observations)})
    if remaining_iteration or iterated_observations != len(expected):
        issue("iterator_coverage_mismatch", {"missing": len(remaining_iteration), "iterated": iterated_observations})
    return {"schema": "hypermem-source-verification-v1", "created_at": now(), "status": "passed" if not problem_count else "failed",
            "source": str(source), "source_sha256": total_hash.hexdigest(), "source_bytes": offset,
            "fingerprint_scope": "scanned_prefix" if not actual_scan_complete else "full_file",
            "selection_scope": "time_window" if window else "prefix" if limit else "all_selected_types",
            "time_window": window,
            "source_total_bytes": source.stat().st_size, "scan_complete": actual_scan_complete,
            "lines": lines, "blank_lines": blank, "record_types": dict(types), "selected_types": sorted(allowed),
            "expected_observations": len(expected), "checked_raw_records": checked_raw,
            "checked_observations": checked_observations, "iterated_observations": iterated_observations,
            "issue_count": problem_count, "issues": problems,
            "scope": "Every selected original JSONL record equals its canonical raw JSON and observation provenance. No records are imported or changed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--allow-prefix", action="store_true", help="Explicitly verify a limited import, not the full source file")
    parser.add_argument("--allow-window", action="store_true", help="Explicitly verify only the recorded time window")
    args = parser.parse_args()
    result = verify_source(args.run, allow_prefix=args.allow_prefix, allow_window=args.allow_window)
    from hypermem.online.io import write_json
    write_json(args.run / "source_verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
