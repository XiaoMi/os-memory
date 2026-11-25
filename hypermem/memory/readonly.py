# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""Read-only access to a HyperOS-Mem database (no migrations, no writes)."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from hypermem.memory.event_memory import EventMemory


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def readonly(run):
    path = Path(run).resolve() / "memory.sqlite"
    if not path.is_file():
        raise ValueError("Run does not contain memory.sqlite")
    conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, isolation_level=None, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA query_only=ON")
        # Connection-local cache only: no protected database setting is changed.
        # Full provenance checks revisit large parent-ID indexes repeatedly.
        conn.execute("PRAGMA cache_size=-65536")
        if conn.execute("PRAGMA application_id").fetchone()[0] != 0x48594D45:
            raise ValueError("Not a Hyper-Mem database")
        if conn.execute("PRAGMA user_version").fetchone()[0] != 2:
            raise ValueError("Requires OS-Memory schema v2; read-only access never migrates a database")
        conn.execute("BEGIN")  # All findings refer to one consistent WAL snapshot.
        yield conn
    finally:
        conn.close()


def memory_view(connection):
    if not connection.execute("PRAGMA query_only").fetchone()[0]:
        raise ValueError("Read-only adapter requires SQLite query_only")
    memory = object.__new__(EventMemory)
    memory._connection = connection
    memory.path = "read-only-view"
    memory._savepoint = 0
    return memory
