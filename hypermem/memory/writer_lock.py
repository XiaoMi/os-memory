# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""Cooperative, fail-fast writer exclusion for Linux/WSL SQLite databases.

Lock a canonical-path sidecar, never open/close SQLite's inode behind its back:
doing so could release process-scoped SQLite byte-range locks. Symlinks resolve
to one sidecar; hardlinked databases are refused (their WAL names are ambiguous).
Readers and independent mode=ro auditors remain usable. This does not constrain
old binaries or arbitrary SQL clients that do not participate in this protocol.
"""
from contextlib import contextmanager
from functools import wraps
import os
from pathlib import Path
import threading


class WriterBusyError(RuntimeError):
    pass


_guard = threading.Lock()
_held = {}


def _after_fork():
    global _guard, _held
    # Close inherited references, never LOCK_UN a parent's open description.
    for entry in _held.values():
        os.close(entry['fd'])
    _held = {}
    _guard = threading.Lock()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_after_fork)


@contextmanager
def writer_lock(memory_or_path):
    path = str(getattr(memory_or_path, 'path', memory_or_path))
    if path == ':memory:':
        yield
        return
    if path == 'read-only-view' and hasattr(memory_or_path, '_connection'):
        if not memory_or_path._connection.execute('PRAGMA query_only').fetchone()[0]:
            raise ValueError('Read-only memory adapter is not query_only')
        yield
        return
    import fcntl
    thread = threading.get_ident()
    key = getattr(memory_or_path, '_writer_identity', None)
    acquired = False
    with _guard:
        entry = _held.get(key)
        if entry:
            if entry['thread'] != thread:
                raise WriterBusyError('Another memory writer is already active for this database')
            entry['depth'] += 1
            acquired = True
    if not acquired:
        database = Path(path).resolve()
        if database.exists() and database.stat().st_nlink > 1:
            raise ValueError('Hardlinked SQLite databases are unsafe; use the original path or a symlink')
        sidecar = database.with_name(database.name + '.writer.lock')
        fd = os.open(sidecar, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            info = os.fstat(fd)
            key = (info.st_dev, info.st_ino)
            with _guard:
                entry = _held.get(key)
                if entry:
                    if entry['thread'] != thread:
                        raise WriterBusyError('Another memory writer is already active for this database')
                    entry['depth'] += 1
                else:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        raise WriterBusyError('Another memory writer is already active for this database') from None
                    _held[key] = {'fd': fd, 'thread': thread, 'depth': 1}
                    fd = None
            if hasattr(memory_or_path, 'path'):
                memory_or_path._writer_identity = key
        finally:
            if fd is not None:
                os.close(fd)
    try:
        yield
    finally:
        with _guard:
            entry = _held[key]
            entry['depth'] -= 1
            if not entry['depth']:
                try:
                    fcntl.flock(entry['fd'], fcntl.LOCK_UN)
                finally:
                    os.close(entry['fd'])
                    del _held[key]


def memory_writer(function):
    @wraps(function)
    def guarded(memory, *args, **kwargs):
        with writer_lock(memory):
            return function(memory, *args, **kwargs)
    return guarded


def engine_writer(function):
    @wraps(function)
    def guarded(engine, *args, **kwargs):
        memory = (args[0] if args else kwargs['memory']) if function.__name__ == '__init__' else engine.memory
        with writer_lock(memory):
            return function(engine, *args, **kwargs)
    return guarded
