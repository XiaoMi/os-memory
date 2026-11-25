# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import fields, replace
from datetime import datetime, timedelta, timezone
import unittest

from hypermem.schema import (
    AtomicObservation, CanonicalEvent, EventRelation, ObservationFeatures, Provenance, TimeRange,
)


NOW = datetime(2026, 9, 8, 10, 0, tzinfo=timezone.utc)


def observation(oid="O1", raw_id="R1", timestamp=NOW):
    return AtomicObservation(oid, timestamp, "test.app", "Booking confirmed", {"order_id": ["ABC"]}, Provenance(raw_id))


def event(eid="E1", observation_ids=None, start=NOW, end=NOW):
    return CanonicalEvent(eid, start, end, "flight booked", ["Test Airline"], {"order_id": ["ABC"]},
                          observation_ids if observation_ids is not None else ["O1"], 0.9)


class SchemaTests(unittest.TestCase):
    def test_observation_has_five_intrinsic_fields_plus_identifier(self):
        self.assertEqual({item.name for item in fields(AtomicObservation)},
                         {"observation_id", "timestamp", "source", "content", "anchors", "provenance"})
        with self.assertRaises(TypeError):
            AtomicObservation.from_dict({**observation().to_dict(), "embedding": [0.1]})
        with self.assertRaises(TypeError):
            AtomicObservation.from_dict({**observation().to_dict(), "entities": ["person"]})

    def test_all_models_json_round_trip(self):
        provenance = Provenance("R1", "raw/input.jsonl", "notification", {"line_number": 1, "nested": [None, True, {"x": "中文"}]})
        models = [provenance, observation(), replace(observation(), provenance=provenance),
                  TimeRange(NOW, NOW + timedelta(minutes=5)),
                  replace(observation(), timestamp=TimeRange(NOW, NOW + timedelta(minutes=5))),
                  event(), replace(event(), embedding=[0.1, -0.2]),
                  ObservationFeatures([0.1, 0.2], ["test"], {"model": "fixture", "revision": "1"}),
                  EventRelation("E1", "E2", "before", 0.8)]
        for model in models:
            with self.subTest(model=type(model).__name__):
                self.assertEqual(type(model).from_json(model.to_json()), model)

    def test_naive_timestamps_rejected_and_offsets_normalized(self):
        with self.assertRaises(ValueError):
            observation(timestamp=datetime(2026, 9, 8))
        shifted = NOW.astimezone(timezone(timedelta(hours=8)))
        self.assertEqual(observation(timestamp=shifted).to_json(), observation().to_json())
        self.assertEqual(observation().start_time, NOW)
        self.assertEqual(observation().end_time, NOW)

    def test_invalid_time_ranges_rejected(self):
        with self.assertRaises(ValueError):
            TimeRange(NOW, NOW - timedelta(seconds=1))
        with self.assertRaises(ValueError):
            event(end=NOW - timedelta(seconds=1))

    def test_invalid_confidence_and_embedding_rejected(self):
        for value in (-0.1, 1.1, float("nan"), float("inf"), True, "0.5"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(event(), confidence=value)
        for value in ([], [float("nan")], [float("inf")], [True], ["1"], "vector"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ObservationFeatures(embedding=value)

    def test_invalid_evidence_anchors_and_json_rejected(self):
        for values in ([], ["O1", "O1"], [""], "O1"):
            with self.subTest(values=values), self.assertRaises(ValueError):
                event(observation_ids=values)
        with self.assertRaises(ValueError):
            replace(observation(), anchors={"order_id": "ABC"})
        with self.assertRaises(ValueError):
            Provenance("R1", metadata={"invalid": float("nan")})
        with self.assertRaises(ValueError):
            Provenance("R1", metadata={1: "invalid key"})

    def test_relations_require_distinct_events_and_supported_type(self):
        for args in (("E1", "E1", "before", 0.5), ("E1", "E2", "invented", 0.5), ("E1", "E2", "before", 2)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                EventRelation(*args)


if __name__ == "__main__":
    unittest.main()
