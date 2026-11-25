# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import sqlite3
import tempfile
import unittest

from hypermem import EventMemory, EventRelation, ObservationFeatures, TimeRange
from test_schema import NOW, event, observation


class EventMemoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "memory.sqlite"
        self.memory = EventMemory(self.path)
        self.addCleanup(self.memory.close)
        self.raw = {"type": "fixture", "payload": {"content": "test", "image_paths": ["image.png"]}}
        self.memory.add_observation(observation(), self.raw)

    def test_create_update_get_list_and_persistence(self):
        self.memory.add_event(event())
        later = NOW + timedelta(minutes=10)
        self.memory.add_observation(observation("O2", "R2", later), {"content": "ticket issued"})
        updated = replace(event(end=later, observation_ids=["O1", "O2"]), state="ticket issued", confidence=0.95)
        self.memory.update_event(updated)
        with EventMemory(self.path) as reopened:
            self.assertEqual(reopened.get_event("E1"), updated)
            self.assertEqual(reopened.list_events(), [updated])
            self.assertEqual([o.observation_id for o in reopened.get_observations("E1")], ["O1", "O2"])
            self.assertEqual(reopened.resolve_provenance("O1"), self.raw)

    def test_trace_recovers_raw_without_original_file(self):
        original = Path(self.directory.name) / "original.jsonl"
        original.write_text('{"message":"test"}\n', encoding="utf-8")
        obs = observation("O-file", "R-file")
        obs = replace(obs, provenance=replace(obs.provenance, raw_path=str(original), metadata={"line_number": 1}))
        self.memory.add_observation(obs, {"message": "test"})
        self.memory.add_event(event("E-file", ["O-file"]))
        original.unlink()
        trace = self.memory.trace_event("E-file")
        self.assertEqual(trace[0]["raw_record"], {"message": "test"})
        self.assertEqual(trace[0]["observation"]["provenance"]["metadata"]["line_number"], 1)

    def test_raw_and_observation_idempotency_and_collision(self):
        self.assertFalse(self.memory.add_observation(observation(), self.raw))
        with self.assertRaises(ValueError):
            self.memory.add_observation(observation(), {"different": "raw"})
        with self.assertRaises(ValueError):
            self.memory.add_observation(replace(observation(), content="changed"), self.raw)
        self.assertEqual(self.memory.get_observation("O1"), observation())
        self.assertEqual(self.memory.get_raw_record("R1"), self.raw)

    def test_failed_observation_collision_rolls_back_new_raw(self):
        changed = observation(raw_id="R-new")
        with self.assertRaises(ValueError):
            self.memory.add_observation(changed, {"new": "raw"})
        with self.assertRaises(KeyError):
            self.memory.get_raw_record("R-new")

    def test_missing_raw_observation_or_event_references_rejected(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.memory.add_observation(observation("O2", "R-missing"))
        with self.assertRaises(KeyError):
            self.memory.add_event(event(observation_ids=["O-missing"]))
        with self.assertRaises(sqlite3.IntegrityError):
            self.memory.set_features("O-missing", ObservationFeatures())
        with self.assertRaises(sqlite3.IntegrityError):
            self.memory.add_relation(EventRelation("E1", "E2", "before", 0.9))
        self.assertEqual(self.memory.counts()["events"], 0)

    def test_features_separate_replaceable_and_persisted(self):
        before = self.memory.get_observation("O1").to_json()
        self.assertIsNone(self.memory.get_features("O1"))
        self.memory.set_features("O1", ObservationFeatures([0.1, 0.2], ["entity"], {"model": "first"}))
        features = ObservationFeatures([0.2, 0.4, 0.6], [], {"model": "second"})
        self.memory.set_features("O1", features)
        with EventMemory(self.path) as reopened:
            self.assertEqual(reopened.get_features("O1"), features)
            self.assertEqual(reopened.get_observation("O1").to_json(), before)

    def test_event_update_cannot_silently_drop_evidence(self):
        self.memory.add_observation(observation("O2", "R2"), {"second": True})
        self.memory.add_event(event(observation_ids=["O1", "O2"]))
        with self.assertRaises(ValueError):
            self.memory.update_event(event())
        self.assertEqual(self.memory.get_event("E1").observation_ids, ["O1", "O2"])

    def test_event_range_must_cover_observation_interval(self):
        self.memory.add_observation(observation("O2", "R2", TimeRange(NOW, NOW + timedelta(hours=1))), {"second": True})
        with self.assertRaises(ValueError):
            self.memory.add_event(event(observation_ids=["O2"]))
        self.memory.add_event(event(observation_ids=["O2"], end=NOW + timedelta(hours=1)))

    def test_observation_cannot_belong_to_two_canonical_events(self):
        self.memory.add_event(event())
        with self.assertRaises(sqlite3.IntegrityError):
            self.memory.add_event(event("E2"))
        with self.assertRaises(KeyError):
            self.memory.get_event("E2")
        self.assertEqual(self.memory.get_observations("E1"), [observation()])

    def test_failed_update_preserves_original_event_and_links(self):
        self.memory.add_event(event())
        self.memory.add_observation(observation("O2", "R2"), {"second": True})
        self.memory.add_event(event("E2", ["O2"]))
        with self.assertRaises(sqlite3.IntegrityError):
            self.memory.update_event(replace(event(observation_ids=["O1", "O2"]), state="changed"))
        self.assertEqual(self.memory.get_event("E1"), event())
        self.assertEqual(self.memory.get_observations("E1"), [observation()])

    def test_relation_crud_and_delete_preserves_source_evidence(self):
        self.memory.add_event(event())
        self.memory.add_observation(observation("O2", "R2"), {"second": True})
        self.memory.add_event(event("E2", ["O2"]))
        relation = EventRelation("E1", "E2", "before", 0.8)
        self.memory.add_relation(relation)
        updated = replace(relation, score=0.95)
        self.memory.add_relation(updated)
        self.assertEqual(self.memory.list_relations("E2"), [updated])
        self.memory.delete_relation("E1", "E2", "before")
        self.assertEqual(self.memory.list_relations(), [])
        self.memory.add_relation(relation)
        self.memory.delete_event("E1")
        self.assertEqual(self.memory.list_relations(), [])
        self.assertEqual(self.memory.resolve_provenance("O1"), self.raw)
        self.assertEqual(len(self.memory.list_events()), 1)

    def test_outer_batch_rollback_includes_nested_writes(self):
        with self.assertRaises(RuntimeError):
            with self.memory.transaction():
                self.memory.add_observation(observation("O2", "R2"), {"second": True})
                self.memory.add_event(event())
                raise RuntimeError("simulated interrupted batch")
        self.assertEqual(self.memory.counts(), {"raw_records": 1, "observations": 1, "observation_features": 0,
                                                "events": 0, "event_relations": 0})

    def test_snapshots_do_not_alias_caller_objects(self):
        self.memory.add_event(event())
        loaded = self.memory.get_event("E1")
        loaded.anchors["order_id"].append("changed")
        self.assertEqual(self.memory.get_event("E1").anchors, {"order_id": ["ABC"]})
        loaded.observation_ids.clear()
        with self.assertRaises(ValueError):
            self.memory.update_event(loaded)

    def test_iter_observations_orders_by_time(self):
        self.memory.add_observation(observation("O2", "R2", NOW - timedelta(days=1)), {"earlier": True})
        self.assertEqual([o.observation_id for o in self.memory.iter_observations()], ["O2", "O1"])

    def test_refuses_unrelated_database(self):
        path = Path(self.directory.name) / "other.sqlite"
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE user_data (id INTEGER)")
        with self.assertRaises(ValueError):
            EventMemory(path)
        with sqlite3.connect(path) as connection:
            self.assertEqual(connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall(), [("user_data",)])

    def test_explicit_create_update_and_missing_ids(self):
        with self.assertRaises(KeyError):
            self.memory.update_event(event())
        self.memory.add_event(event())
        with self.assertRaises(ValueError):
            self.memory.add_event(event())
        for operation in (self.memory.get_event, self.memory.get_observations, self.memory.delete_event,
                          self.memory.get_observation, self.memory.get_raw_record, self.memory.get_features):
            with self.subTest(operation=operation.__name__), self.assertRaises(KeyError):
                operation("missing")


class DatabaseFileModeTests(unittest.TestCase):
    def test_new_database_is_owner_only(self):
        import os, stat
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.db"
            from hypermem import HyperMemory
            with HyperMemory(path) as memory:
                memory.add([{"type": "sms", "timestamp": "2026-05-03 08:00",
                             "data": {"sender": "10086", "content": "synthetic test message"}}], user="U01")
            for candidate in Path(directory).iterdir():
                if candidate.name.startswith("memory.db") and not candidate.name.endswith(".writer.lock"):
                    self.assertEqual(stat.S_IMODE(os.stat(candidate).st_mode), 0o600, candidate.name)

    def test_existing_database_mode_is_kept(self):
        import os, stat
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.db"
            EventMemory(path).close()
            os.chmod(path, 0o640)
            EventMemory(path).close()
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o640)


if __name__ == "__main__":
    unittest.main()
