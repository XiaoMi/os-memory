# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

import unittest

from hypermem.api import _shape_result


def retrieved(*events):
    """Retriever output: events listed by start time, with their scores."""
    return {"events": [{"event_id": eid, "title": eid, "span": [start, start], "facts": []}
                       for eid, start, _ in events],
            "node_scores": {eid: score for eid, _, score in events}, "long_memories": []}


class SearchResultTests(unittest.TestCase):
    def test_keeps_the_most_relevant_events_in_time_order(self):
        result = _shape_result("q", retrieved(("E1", "2026-05-01", 0.1), ("E2", "2026-05-02", 0.9),
                                              ("E3", "2026-05-03", 0.5), ("E4", "2026-05-04", 0.7)), top_k=2)
        self.assertEqual([e["event_id"] for e in result["events"]], ["E2", "E4"])
        self.assertEqual([e["score"] for e in result["events"]], [0.9, 0.7])

    def test_top_k_beyond_the_results_keeps_them_all(self):
        result = _shape_result("q", retrieved(("E1", "2026-05-01", 0.1), ("E2", "2026-05-02", 0.9)), top_k=20)
        self.assertEqual([e["event_id"] for e in result["events"]], ["E1", "E2"])

    def test_equal_scores_keep_the_earlier_event(self):
        result = _shape_result("q", retrieved(("E1", "2026-05-01", 0.5), ("E2", "2026-05-02", 0.5)), top_k=1)
        self.assertEqual([e["event_id"] for e in result["events"]], ["E1"])


if __name__ == "__main__":
    unittest.main()
