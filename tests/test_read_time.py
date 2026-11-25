# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from hypermem import EventMemory, HyperMemory
from hypermem.api import _time_bound
from test_schema import NOW, event, observation


def ms(moment):
    return int(moment.timestamp() * 1000)


class TimeBoundTests(unittest.TestCase):
    def test_dates_and_times(self):
        local = lambda *parts: datetime(*parts).astimezone()
        self.assertEqual(_time_bound("2026-05-03T08:30+08:00", "start"),
                         ms(datetime(2026, 5, 3, 0, 30, tzinfo=timezone.utc)))
        self.assertEqual(_time_bound("2026-05-03T00:30Z", "start"),
                         ms(datetime(2026, 5, 3, 0, 30, tzinfo=timezone.utc)))
        self.assertEqual(_time_bound("2026-05-03 08:30", "start"), ms(local(2026, 5, 3, 8, 30)))
        self.assertEqual(_time_bound("2026-05-03", "start"), ms(local(2026, 5, 3)))
        self.assertEqual(_time_bound(date(2026, 5, 3), "start"), ms(local(2026, 5, 3)))
        aware = datetime(2026, 5, 3, 8, 30, tzinfo=timezone(timedelta(hours=8)))
        self.assertEqual(_time_bound(aware, "start"), ms(aware))
        self.assertEqual(_time_bound(1777766400000, "start"), 1777766400000)
        self.assertIsNone(_time_bound(None, "start"))

    def test_an_end_date_includes_the_whole_day(self):
        local = lambda *parts: datetime(*parts).astimezone()
        self.assertEqual(_time_bound("2026-05-03", "end", end=True), ms(local(2026, 5, 4)) - 1)
        self.assertEqual(_time_bound(date(2026, 5, 3), "end", end=True), ms(local(2026, 5, 4)) - 1)
        # A time, not a date alone, is an exact bound.
        self.assertEqual(_time_bound("2026-05-03 00:00", "end", end=True), ms(local(2026, 5, 3)))

    def test_rejects_what_is_not_a_time(self):
        for value in ("yesterday", "2026/05/03", 1.5, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "start must be a date or time"):
                _time_bound(value, "start")


class ReadByTimeTests(unittest.TestCase):
    def test_read_selects_events_by_date(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "memory.db"
        later = NOW + timedelta(days=2)
        with EventMemory(path) as memory:
            memory.add_observation(observation("O1", "R1", NOW), {"content": "first"})
            memory.add_observation(observation("O2", "R2", later), {"content": "second"})
            memory.add_event(event("E1", ["O1"], NOW, NOW))
            memory.add_event(event("E2", ["O2"], later, later))
        with HyperMemory(path) as mem:
            day = NOW.astimezone().date().isoformat()
            ids = lambda **filters: [e["event_id"] for e in mem.read(**filters)["events"]]
            self.assertEqual(ids(start=day, end=day), ["E1"])
            self.assertEqual(ids(start=day), ["E1", "E2"])
            self.assertEqual(ids(end=day), ["E1"])
            self.assertEqual(ids(start=later.isoformat()), ["E2"])
            with self.assertRaisesRegex(ValueError, "end must be a date or time"):
                mem.read(end="next week")


if __name__ == "__main__":
    unittest.main()
