# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""SQLite Event Memory: one authoritative store, with explicit evidence links.

No localization, event identity inference, or model calls occur in this layer.
Writes use savepoints, so a caller can batch operations in one transaction.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
import json

from hypermem.schema import AtomicObservation, CanonicalEvent, EventRelation, ObservationFeatures
from hypermem.schema.serialization import encode_json, require_text
from hypermem.memory.writer_lock import memory_writer


_APPLICATION_ID = 0x48594D45  # HYME
_SCHEMA_VERSION = 2
_PIPELINE_TABLES = (
    "CREATE TABLE event_changes (sequence INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL)",
    "CREATE TRIGGER event_added AFTER INSERT ON events BEGIN INSERT INTO event_changes(event_id) VALUES (NEW.event_id); END",
    "CREATE TRIGGER event_updated AFTER UPDATE ON events BEGIN INSERT INTO event_changes(event_id) VALUES (NEW.event_id); END",
    "CREATE TRIGGER event_deleted AFTER DELETE ON events BEGIN INSERT INTO event_changes(event_id) VALUES (OLD.event_id); END",
    "CREATE TABLE pipeline_metadata (key TEXT PRIMARY KEY, value_json TEXT NOT NULL)",
    "CREATE TABLE decision_log (sequence INTEGER PRIMARY KEY AUTOINCREMENT, observation_id TEXT NOT NULL REFERENCES observations(observation_id), audit_json TEXT NOT NULL)",
    "CREATE INDEX decisions_observation ON decision_log(observation_id)",
    "CREATE TABLE pending_observations (observation_id TEXT PRIMARY KEY REFERENCES observations(observation_id), attempts INTEGER NOT NULL DEFAULT 1, rationale TEXT NOT NULL)",
)
_TABLES = (
    """CREATE TABLE raw_records (
        raw_id TEXT PRIMARY KEY,
        raw_json TEXT NOT NULL
    )""",
    """CREATE TABLE observations (
        observation_id TEXT PRIMARY KEY,
        raw_id TEXT NOT NULL REFERENCES raw_records(raw_id),
        start_time TEXT NOT NULL,
        end_time TEXT NOT NULL,
        observation_json TEXT NOT NULL
    )""",
    "CREATE INDEX observations_time ON observations(start_time, observation_id)",
    """CREATE TABLE observation_features (
        observation_id TEXT PRIMARY KEY REFERENCES observations(observation_id) ON DELETE CASCADE,
        features_json TEXT NOT NULL
    )""",
    """CREATE TABLE events (
        event_id TEXT PRIMARY KEY,
        start_time TEXT NOT NULL,
        end_time TEXT NOT NULL,
        event_json TEXT NOT NULL
    )""",
    "CREATE INDEX events_time ON events(start_time, event_id)",
    """CREATE TABLE event_observations (
        event_id TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
        observation_id TEXT NOT NULL UNIQUE REFERENCES observations(observation_id),
        position INTEGER NOT NULL,
        PRIMARY KEY(event_id, observation_id),
        UNIQUE(event_id, position)
    )""",
    """CREATE TABLE event_relations (
        source_event TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
        target_event TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
        relation_type TEXT NOT NULL,
        score REAL NOT NULL CHECK(score >= 0 AND score <= 1),
        PRIMARY KEY(source_event, target_event, relation_type)
    )""",
    "CREATE INDEX relations_target ON event_relations(target_event)",
)


class EventMemory:
    """Local SQLite implementation of the milestone 1–2 persistence interface.

    Raw records and observations are immutable: exact reimports are idempotent,
    while ID collisions with different content raise ValueError. Event updates
    retain existing evidence. Explicit deletion removes an event and its links,
    but retains the raw records, observations, and derived features.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path) if str(path) == ':memory:' else str(Path(path).resolve())
        from hypermem.memory.writer_lock import writer_lock
        with writer_lock(self):
            self._open()

    def _open(self):
        if self.path != ':memory:':
            # Personal data: create a new database owner-only. SQLite gives its
            # -wal and -shm files the same mode as the database file.
            try:
                os.close(os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600))
            except FileExistsError:
                pass
        self._connection = sqlite3.connect(self.path, isolation_level=None, timeout=30)
        self._connection.row_factory = sqlite3.Row
        self._savepoint = 0
        try:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._initialize_schema()
            self._connection.execute("CREATE INDEX IF NOT EXISTS events_interval_end ON events(end_time,start_time,event_id)")
        except Exception:
            self.close()
            raise

    def _initialize_schema(self) -> None:
        application = self._connection.execute("PRAGMA application_id").fetchone()[0]
        version = self._connection.execute("PRAGMA user_version").fetchone()[0]
        tables = self._connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        if application == _APPLICATION_ID and version == _SCHEMA_VERSION:
            return
        if application == _APPLICATION_ID and version == 1:
            with self.transaction():
                for statement in _PIPELINE_TABLES:
                    self._connection.execute(statement)
                self._connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            return
        if application != 0 or version != 0 or tables:
            raise ValueError("Not an empty database or a supported Hyper-Mem schema")
        with self.transaction():
            for statement in _TABLES:
                self._connection.execute(statement)
            for statement in _PIPELINE_TABLES:
                self._connection.execute(statement)
            self._connection.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
            self._connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")

    @contextmanager
    def transaction(self) -> Iterator[EventMemory]:
        """Atomic batch; nested method calls use their own rollback savepoints."""
        from hypermem.memory.writer_lock import writer_lock
        with writer_lock(self):
            with self._transaction_unlocked():
                yield self

    @contextmanager
    def _transaction_unlocked(self):
        self._savepoint += 1
        name = f"hypermem_{self._savepoint}"
        self._connection.execute(f"SAVEPOINT {name}")
        try:
            yield self
        except BaseException:
            self._connection.execute(f"ROLLBACK TO {name}")
            self._connection.execute(f"RELEASE {name}")
            raise
        else:
            self._connection.execute(f"RELEASE {name}")

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> EventMemory:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def add_raw_record(self, raw_id: str, raw_record: dict[str, Any]) -> bool:
        require_text(raw_id, "raw_id")
        if not isinstance(raw_record, dict):
            raise ValueError("raw_record must be a JSON object")
        encoded = encode_json(raw_record)
        with self.transaction():
            row = self._connection.execute("SELECT raw_json FROM raw_records WHERE raw_id=?", (raw_id,)).fetchone()
            if row is not None:
                if row[0] != encoded:
                    raise ValueError("raw_id collision with different content")
                return False
            self._connection.execute("INSERT INTO raw_records VALUES (?, ?)", (raw_id, encoded))
        return True

    def get_raw_record(self, raw_id: str) -> dict[str, Any]:
        row = self._connection.execute("SELECT raw_json FROM raw_records WHERE raw_id=?", (raw_id,)).fetchone()
        if row is None:
            raise KeyError(raw_id)
        return json.loads(row[0])

    def add_observation(self, observation: AtomicObservation, raw_record: dict[str, Any] | None = None) -> bool:
        # Revalidation catches mutations of nested lists/dicts in frozen dataclasses.
        observation = AtomicObservation.from_dict(observation.to_dict())
        encoded = observation.to_json()
        with self.transaction():
            if raw_record is not None:
                self.add_raw_record(observation.provenance.raw_id, raw_record)
            row = self._connection.execute(
                "SELECT observation_json FROM observations WHERE observation_id=?", (observation.observation_id,)
            ).fetchone()
            if row is not None:
                if row[0] != encoded:
                    raise ValueError("observation_id collision with different content or provenance")
                return False
            self._connection.execute(
                "INSERT INTO observations VALUES (?, ?, ?, ?, ?)",
                (observation.observation_id, observation.provenance.raw_id,
                 observation.start_time.isoformat(), observation.end_time.isoformat(), encoded),
            )
        return True

    def get_observation(self, observation_id: str) -> AtomicObservation:
        row = self._connection.execute(
            "SELECT observation_json FROM observations WHERE observation_id=?", (observation_id,)
        ).fetchone()
        if row is None:
            raise KeyError(observation_id)
        return AtomicObservation.from_json(row[0])

    def iter_observations(self) -> Iterator[AtomicObservation]:
        """Chronological keyset pages; no read statement spans formation commits.

        Inputs are immutable and imported before offline replay. Finishing each
        SELECT before yielding lets WAL checkpoints advance between observations.
        A single cursor over the entire input pins the WAL for the whole replay.
        """
        after = None
        while True:
            if after is None:
                rows = self._connection.execute(
                    "SELECT observation_json,start_time,observation_id FROM observations ORDER BY start_time,observation_id LIMIT 512"
                ).fetchall()
            else:
                rows = self._connection.execute(
                    "SELECT observation_json,start_time,observation_id FROM observations WHERE (start_time,observation_id)>(?,?) ORDER BY start_time,observation_id LIMIT 512",
                    after,
                ).fetchall()
            if not rows:
                break
            after = (rows[-1][1], rows[-1][2])
            for row in rows:
                yield AtomicObservation.from_json(row[0])

    def set_features(self, observation_id: str, features: ObservationFeatures) -> None:
        features = ObservationFeatures.from_dict(features.to_dict())
        with self.transaction():
            self._connection.execute(
                """INSERT INTO observation_features VALUES (?, ?)
                ON CONFLICT(observation_id) DO UPDATE SET features_json=excluded.features_json""",
                (observation_id, features.to_json()),
            )

    def get_features(self, observation_id: str) -> ObservationFeatures | None:
        self.get_observation(observation_id)
        row = self._connection.execute(
            "SELECT features_json FROM observation_features WHERE observation_id=?", (observation_id,)
        ).fetchone()
        return ObservationFeatures.from_json(row[0]) if row is not None else None

    def add_event(self, event: CanonicalEvent) -> None:
        self._write_event(event, update=False)

    def update_event(self, event: CanonicalEvent) -> None:
        self._write_event(event, update=True)

    def _write_event(self, event: CanonicalEvent, *, update: bool) -> None:
        event = CanonicalEvent.from_dict(event.to_dict())
        with self.transaction():
            old = self._connection.execute("SELECT event_json FROM events WHERE event_id=?", (event.event_id,)).fetchone()
            if update:
                if old is None:
                    raise KeyError(event.event_id)
                previous = CanonicalEvent.from_json(old[0])
                if not set(previous.observation_ids).issubset(event.observation_ids):
                    raise ValueError("event updates must preserve all existing supporting observations")
            elif old is not None:
                raise ValueError("event_id already exists; use update_event")
            for observation_id in event.observation_ids:
                observation = self.get_observation(observation_id)
                if observation.start_time < event.start_time or observation.end_time > event.end_time:
                    raise ValueError("event time range must cover every supporting observation")
            self._connection.execute(
                """INSERT INTO events VALUES (?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET start_time=excluded.start_time,
                    end_time=excluded.end_time, event_json=excluded.event_json""",
                (event.event_id, event.start_time.isoformat(), event.end_time.isoformat(), event.to_json()),
            )
            self._connection.execute("DELETE FROM event_observations WHERE event_id=?", (event.event_id,))
            self._connection.executemany(
                "INSERT INTO event_observations VALUES (?, ?, ?)",
                ((event.event_id, observation_id, position) for position, observation_id in enumerate(event.observation_ids)),
            )

    def get_event(self, event_id: str) -> CanonicalEvent:
        row = self._connection.execute("SELECT event_json FROM events WHERE event_id=?", (event_id,)).fetchone()
        if row is None:
            raise KeyError(event_id)
        return CanonicalEvent.from_json(row[0])

    def list_events(self) -> list[CanonicalEvent]:
        return list(self.iter_events())

    def iter_events(self, *, include_embedding: bool = True) -> Iterator[CanonicalEvent]:
        for row in self._connection.execute("SELECT event_json FROM events ORDER BY start_time, event_id"):
            value = json.loads(row[0])
            if not include_embedding:
                value["embedding"] = None
            yield CanonicalEvent.from_dict(value)

    def delete_event(self, event_id: str) -> None:
        with self.transaction():
            cursor = self._connection.execute("DELETE FROM events WHERE event_id=?", (event_id,))
            if cursor.rowcount == 0:
                raise KeyError(event_id)

    def get_observations(self, event_id: str) -> list[AtomicObservation]:
        self.get_event(event_id)
        return [AtomicObservation.from_json(row[0]) for row in self._connection.execute(
            """SELECT o.observation_json FROM observations o
            JOIN event_observations e ON o.observation_id=e.observation_id
            WHERE e.event_id=? ORDER BY e.position""", (event_id,)
        )]

    def resolve_provenance(self, observation_id: str) -> dict[str, Any]:
        return self.get_raw_record(self.get_observation(observation_id).provenance.raw_id)

    def trace_event(self, event_id: str) -> list[dict[str, Any]]:
        return [{"observation": observation.to_dict(), "raw_record": self.get_raw_record(observation.provenance.raw_id)}
                for observation in self.get_observations(event_id)]

    def add_relation(self, relation: EventRelation) -> None:
        relation = EventRelation.from_dict(relation.to_dict())
        with self.transaction():
            self._connection.execute(
                """INSERT INTO event_relations VALUES (?, ?, ?, ?)
                ON CONFLICT(source_event, target_event, relation_type) DO UPDATE SET score=excluded.score""",
                (relation.source_event, relation.target_event, relation.relation_type, relation.score),
            )

    def list_relations(self, event_id: str | None = None) -> list[EventRelation]:
        parameters: tuple[str, ...] = ()
        query = "SELECT * FROM event_relations"
        if event_id is not None:
            self.get_event(event_id)
            query += " WHERE source_event=? OR target_event=?"
            parameters = (event_id, event_id)
        query += " ORDER BY source_event, target_event, relation_type"
        return [EventRelation(**dict(row)) for row in self._connection.execute(query, parameters)]

    def delete_relation(self, source_event: str, target_event: str, relation_type: str) -> None:
        with self.transaction():
            cursor = self._connection.execute(
                "DELETE FROM event_relations WHERE source_event=? AND target_event=? AND relation_type=?",
                (source_event, target_event, relation_type),
            )
            if cursor.rowcount == 0:
                raise KeyError((source_event, target_event, relation_type))

    def counts(self) -> dict[str, int]:
        return {table: self._connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("raw_records", "observations", "observation_features", "events", "event_relations")}

    def get_metadata(self, key: str, default: Any = None) -> Any:
        row = self._connection.execute("SELECT value_json FROM pipeline_metadata WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_metadata(self, key: str, value: Any) -> None:
        from hypermem.memory.writer_lock import writer_lock
        with writer_lock(self):
            self._connection.execute("INSERT INTO pipeline_metadata VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json",
                                     (key, encode_json(value)))

    @memory_writer
    def log_decision(self, observation_id: str, audit: dict[str, Any]) -> None:
        self._connection.execute("INSERT INTO decision_log(observation_id, audit_json) VALUES (?, ?)",
                                 (observation_id, encode_json(audit)))

    def iter_decisions(self) -> Iterator[dict[str, Any]]:
        for row in self._connection.execute("SELECT audit_json FROM decision_log ORDER BY sequence"):
            yield json.loads(row[0])

    def last_decision(self, observation_id: str) -> dict[str, Any] | None:
        row = self._connection.execute("SELECT audit_json FROM decision_log WHERE observation_id=? ORDER BY sequence DESC LIMIT 1",
                                       (observation_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def event_for_observation(self, observation_id: str) -> str | None:
        row = self._connection.execute("SELECT event_id FROM event_observations WHERE observation_id=?", (observation_id,)).fetchone()
        return row[0] if row else None

    @memory_writer
    def defer(self, observation_id: str, rationale: str) -> None:
        self._connection.execute("""INSERT INTO pending_observations VALUES (?, 1, ?)
            ON CONFLICT(observation_id) DO UPDATE SET attempts=attempts+1, rationale=excluded.rationale""", (observation_id, rationale))

    @memory_writer
    def clear_pending(self, observation_id: str) -> None:
        self._connection.execute("DELETE FROM pending_observations WHERE observation_id=?", (observation_id,))

    def list_pending(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._connection.execute("SELECT * FROM pending_observations ORDER BY observation_id")]
