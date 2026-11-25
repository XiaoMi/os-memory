# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest

from hypermem import EventMemory, Provenance, TimeRange
from hypermem.adapters import OSTraceAdapter
from hypermem.import_observations import import_jsonl, main


def raw_record(kind="real", ts=1788852000123, **data):
    return {"ts": ts, "ts_end": ts, "type": kind, "payload": {
        "src_line": 42, "user_id": "synthetic-test-user", "bucket": "fixture", "image_paths": ["fixture.png"],
        "event": {"from_app": "Test App", "from_pkg": "test.app", "from_device": "fixture",
                  "perception_type": "notification", "active_name": "notify", "active_object": "fixture",
                  "data": {"content": "测试通知", **data}},
    }}


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.input = self.root / "fixture.jsonl"
        self.memory = EventMemory(self.root / "memory.sqlite")
        self.addCleanup(self.memory.close)

    def write_records(self, records):
        self.input.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n", encoding="utf-8")

    def test_adapter_preserves_source_content_interval_and_explicit_anchors(self):
        raw = raw_record(order_id="order-A")
        raw["ts_end"] += 30000
        provenance = Provenance("raw", "fixture.jsonl", "real", {"line_number": 1})
        observation = OSTraceAdapter().parse(raw, observation_id="obs", provenance=provenance)
        self.assertIsInstance(observation.timestamp, TimeRange)
        self.assertEqual(observation.end_time - observation.start_time, timedelta(seconds=30))
        self.assertEqual(observation.start_time, datetime(2026, 9, 8, 7, 20, 0, 123000, tzinfo=timezone.utc))
        self.assertEqual(observation.source, "test.app")
        self.assertEqual(json.loads(observation.content), raw["payload"]["event"])
        self.assertEqual(observation.anchors, {"order_id": ["order-A"]})
        self.assertEqual(observation.provenance, provenance)

    def test_default_filters_summaries_and_synthetic_records(self):
        records = [raw_record(), {"type": "segment", "payload": {"description": "existing summary"}},
                   raw_record("wifi_segment"), raw_record("gps_segment"), raw_record(ts=1788852001123)]
        self.write_records(records)
        report = import_jsonl(self.input, source_id="fixture", memory=self.memory)
        self.assertEqual(report["accepted_observations"], 2)
        self.assertEqual(report["skipped_types"], {"gps_segment": 1, "segment": 1, "wifi_segment": 1})
        self.assertFalse(report["event_formation_performed"])
        self.assertEqual(self.memory.counts()["events"], 0)
        for obs in self.memory.iter_observations():
            line = obs.provenance.metadata["line_number"]
            self.assertEqual(self.memory.resolve_provenance(obs.observation_id), records[line - 1])
            self.assertIsNone(self.memory.get_features(obs.observation_id))

    def test_synthetic_location_requires_opt_in_and_retains_type(self):
        self.write_records([raw_record("wifi_segment"), raw_record("gps_segment")])
        report = import_jsonl(self.input, source_id="fixture", memory=self.memory, include_synthetic_location=True)
        self.assertEqual(report["accepted_observations"], 2)
        self.assertEqual({o.provenance.source_type for o in self.memory.iter_observations()}, {"wifi_segment", "gps_segment"})
        self.assertTrue(all(o.provenance.metadata["synthetic"] for o in self.memory.iter_observations()))

    def test_pmm_cleaned_preserves_original_source_line_mapping(self):
        raw = raw_record()
        raw["_cleaning"] = {"observation_id":"fixture-cleaned", "source_lines":[42, 43, 43], "repeat_count":2}
        self.write_records([raw])
        report = import_jsonl(self.input, source_id="pmm-cleaned", memory=self.memory,
                              input_format="pmm_cleaned", user_id="fixture-user")
        observation = next(self.memory.iter_observations())
        self.assertEqual(report["accepted_observations"], 1)
        self.assertEqual(observation.provenance.metadata["source_lines"], [42, 43])
        self.assertEqual(observation.provenance.metadata["input_format"], "pmm_cleaned")
        self.assertEqual(observation.provenance.metadata["user_id"], "fixture-user")
        self.assertEqual(json.loads(observation.content), raw["payload"]["event"])

    def test_raw_line_pointer_and_sha256_survive_blank_lines_and_bom(self):
        import hashlib
        line = json.dumps(raw_record(), ensure_ascii=False).encode("utf-8") + b"\n"
        self.input.write_bytes(b"\xef\xbb\xbf\n" + line)
        report = import_jsonl(self.input, source_id="fixture", memory=self.memory)
        obs = next(self.memory.iter_observations())
        self.assertEqual(report["blank_lines"], 1)
        self.assertEqual(obs.provenance.metadata["line_number"], 2)
        self.assertEqual(obs.provenance.metadata["byte_offset"], 4)
        self.assertEqual(obs.provenance.metadata["raw_line_sha256"], hashlib.sha256(line).hexdigest())

    def test_same_input_reimport_is_idempotent_and_namespaces_are_distinct(self):
        self.write_records([raw_record(), raw_record()])
        first = import_jsonl(self.input, source_id="fixture", memory=self.memory)
        repeated = import_jsonl(self.input, source_id="fixture", memory=self.memory)
        self.assertEqual(first["inserted_observations"], 2)
        self.assertEqual(repeated["inserted_observations"], 0)
        self.assertEqual(repeated["duplicate_observations"], 2)
        self.assertEqual(first["scanned_bytes_sha256"], repeated["scanned_bytes_sha256"])
        import_jsonl(self.input, source_id="other-fixture", memory=self.memory)
        self.assertEqual(self.memory.counts()["observations"], 4)

    def test_changed_input_cannot_overwrite_existing_provenance(self):
        self.write_records([raw_record()])
        import_jsonl(self.input, source_id="fixture", memory=self.memory)
        self.write_records([raw_record(content="changed")])
        with self.assertRaisesRegex(ValueError, "line 1"):
            import_jsonl(self.input, source_id="fixture", memory=self.memory)
        obs = next(self.memory.iter_observations())
        self.assertEqual(self.memory.resolve_provenance(obs.observation_id), raw_record())

    def test_malformed_later_record_rolls_back_entire_import(self):
        self.input.write_text(json.dumps(raw_record()) + "\ninvalid json\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "line 2"):
            import_jsonl(self.input, source_id="fixture", memory=self.memory)
        self.assertEqual(self.memory.counts()["observations"], 0)
        self.assertEqual(self.memory.counts()["raw_records"], 0)

    def test_duplicate_json_keys_and_non_finite_values_rejected(self):
        for text in ('{"type":"real","type":"segment"}', '{"type":"real","payload":NaN}'):
            self.input.write_text(text + "\n", encoding="utf-8")
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "line 1"):
                import_jsonl(self.input, source_id="fixture")

    def test_limit_counts_accepted_observations_not_skipped_rows(self):
        self.write_records([{"type": "segment"}, raw_record(), raw_record()])
        report = import_jsonl(self.input, source_id="fixture", limit=1)
        self.assertEqual(report["accepted_observations"], 1)
        self.assertEqual(report["lines_scanned"], 2)
        self.assertFalse(report["scan_complete"])
        self.assertEqual(self.memory.counts()["observations"], 0)

    def test_reversed_ranges_bad_timestamps_and_missing_payload_rejected(self):
        bad_range = raw_record()
        bad_range["ts_end"] -= 1
        missing_payload = raw_record()
        del missing_payload["payload"]
        for record in (bad_range, missing_payload, raw_record(ts="2026-09-08"), raw_record(ts=True)):
            self.write_records([record])
            with self.subTest(record_type=type(record["ts"]).__name__), self.assertRaisesRegex(ValueError, "line 1"):
                import_jsonl(self.input, source_id="fixture")

    def test_unsorted_input_is_reported_and_storage_reads_chronologically(self):
        self.write_records([raw_record(ts=1788852001123), raw_record(ts=1788852000123)])
        report = import_jsonl(self.input, source_id="fixture", memory=self.memory)
        self.assertEqual(report["out_of_order_observations"], 1)
        self.assertEqual([o.provenance.metadata["line_number"] for o in self.memory.iter_observations()], [2, 1])

    def test_cli_writes_report_database_and_refuses_existing_output(self):
        self.write_records([raw_record()])
        output = self.root / "run"
        args = ["--input", str(self.input), "--source-id", "fixture", "--output", str(output)]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(args), 0)
        report_path = output / "import_report.json"
        first_report = report_path.read_bytes()
        self.assertEqual(json.loads(first_report)["storage_counts"]["observations"], 1)
        with redirect_stderr(io.StringIO()):
            self.assertEqual(main(args), 1)
        self.assertEqual(report_path.read_bytes(), first_report)

    def test_validate_only_does_not_create_database(self):
        self.write_records([raw_record()])
        output = self.root / "validation"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--input", str(self.input), "--source-id", "fixture", "--validate-only",
                                   "--output", str(output)]), 0)
        self.assertFalse((output / "memory.sqlite").exists())
        self.assertTrue((output / "import_report.json").exists())


if __name__ == "__main__":
    unittest.main()
