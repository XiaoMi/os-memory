# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

LIGHTRAG_SOURCE = Path(__file__).resolve().parents[1] / "systems" / "lightrag" / "source"
if not (LIGHTRAG_SOURCE / "lightrag" / "file_atomic.py").exists():
    raise unittest.SkipTest("LightRAG source not fetched; run python scripts/fetch_baselines.py --systems lightrag")
sys.path.insert(0, str(LIGHTRAG_SOURCE))

from lightrag import file_atomic  # noqa: E402
from lightrag.file_atomic import WINDOWS_REPLACE_RETRIES, atomic_write  # noqa: E402


class WindowsOS:
    """Stands in for the ``os`` module inside file_atomic only: reports ``nt``
    and routes ``replace`` through a fake, delegating everything else to the
    real module. Patching ``os.name`` / ``os.replace`` themselves would change
    them for the whole process (pathlib included)."""

    name = "nt"

    def __init__(self, replace):
        self.replace = replace

    def __getattr__(self, attribute):
        return getattr(os, attribute)


class AtomicWriteTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.destination = Path(tmp.name) / "cache.json"
        sleep = patch.object(file_atomic.time, "sleep", lambda _: None)
        sleep.start()
        self.addCleanup(sleep.stop)

    def test_retries_windows_permission_error(self):
        attempts = 0

        def replace_with_transient_lock(source, target):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise PermissionError("file is temporarily locked")
            os.replace(source, target)

        with patch.object(file_atomic, "os", WindowsOS(replace_with_transient_lock)):
            atomic_write(str(self.destination), lambda path: Path(path).write_text("ok"))

        self.assertEqual(self.destination.read_text(), "ok")
        self.assertEqual(attempts, 3)
        self.assertEqual(list(self.destination.parent.iterdir()), [self.destination])

    def test_propagates_persistent_permission_error(self):
        attempts = 0

        def replace_always_locked(source, target):
            nonlocal attempts
            attempts += 1
            raise PermissionError("locked")

        with patch.object(file_atomic, "os", WindowsOS(replace_always_locked)):
            with self.assertRaises(PermissionError):
                atomic_write(str(self.destination), lambda path: Path(path).write_text("ok"))

        self.assertEqual(attempts, WINDOWS_REPLACE_RETRIES)
        self.assertEqual(list(self.destination.parent.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
