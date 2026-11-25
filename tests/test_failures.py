# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

from pathlib import Path
import tempfile
import unittest

from hypermem.memory import EventMemory
from hypermem.online.consolidation import DDL, save
from hypermem.online.failures import boundary_failures


def task(status, attempts, limit=None):
    value = {"status": status, "attempts": attempts, "input": {"stage": "wiki"}, "error": "bad reply"}
    if limit is not None:
        value["attempt_limit"] = limit
    return value


class BoundaryFailureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.m = EventMemory(Path(self.tmp.name) / "memory.sqlite")
        with self.m.transaction():
            for statement in DDL:
                self.m._connection.execute(statement)
            self.m.set_metadata("config", {"online": {"consolidation": {"max_attempts": 3}}})

    def tearDown(self):
        self.m.close()
        self.tmp.cleanup()

    def finish(self, status, unfinished, *tasks):
        with self.m.transaction():
            for n, t in enumerate(tasks):
                save(self.m, "on_consolidation_tasks", f"T{n}", t)
            self.m.set_metadata("online_consolidation_last", {"status": status, "unfinished_tasks": unfinished})

    def test_complete_run_has_no_skips(self):
        self.finish("complete", 0, task("complete", 1))
        self.assertEqual(boundary_failures(self.m), [])

    def test_exhausted_tasks_are_skipped_and_listed(self):
        self.finish("needs_review", 2, task("complete", 1), task("failed", 3), task("interrupted", 4, limit=4))
        skipped = boundary_failures(self.m)
        self.assertEqual([f["id"] for f in skipped], ["T1", "T2"])
        self.assertEqual(skipped[0]["error"], "bad reply")

    def test_retryable_or_unreported_work_is_not_skipped(self):
        for status, unfinished, tasks in (("needs_review", 1, [task("failed", 2)]),
                                          ("needs_review", 1, [task("pending", 0)]),
                                          ("needs_review", 2, [task("failed", 3)]),
                                          ("paused_not_idle", 0, [])):
            with self.subTest(status=status, tasks=tasks):
                self.m._connection.execute("DELETE FROM on_consolidation_tasks")
                self.finish(status, unfinished, *tasks)
                with self.assertRaises(ValueError):
                    boundary_failures(self.m)


if __name__ == "__main__":
    unittest.main()
