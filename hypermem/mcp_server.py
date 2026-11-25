# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""HyperOS-Mem MCP server (stdio): exposes the HyperMemory library to MCP clients.

The tools mirror the library methods of the same names:

    add(records | jsonl_path, user) HyperMemory.add     save records, inline or from a JSONL file
    build()                        HyperMemory.build    construct the memory graph (DRED + SAGE)
    search(query, top_k, as_text)  HyperMemory.search   semantic search
    read(...)                      HyperMemory.read     structured browse
    status()                       HyperMemory.is_started / is_built / count(), plus the build job

``build`` calls the LLM and can take minutes, longer than an MCP client waits
for a tool call, so the tool starts the build in the background and returns at
once; ``status`` follows it and returns the library's build report when done.
A build stopped by a model service failure is resumed by calling ``build``
again once the service is back.

Run with ``hyperos-mem-mcp`` or ``python -m hypermem.mcp_server`` (requires
``pip install "hyperos-mem[mcp]"``). Configuration comes from environment
variables, normally set in the MCP client's server config:

    HYPERMEM_DB                   absolute path of the memory SQLite file (required;
                                  add creates it, the other tools need it to exist)
    HYPERMEM_EMBEDDING_URL        OpenAI-compatible .../v1/embeddings URL  (search, build)
    HYPERMEM_EMBEDDING_MODEL      embedding model name                     (search, build)
    HYPERMEM_EMBEDDING_DIMENSION  embedding dimension                      (search, build)
    HYPERMEM_EMBEDDING_API_KEY    embedding API key, if the service needs one
    HYPERMEM_LLM_BASE_URL         OpenAI-compatible .../v1 base URL        (build)
    HYPERMEM_LLM_MODEL            LLM model name                           (build)
    HYPERMEM_LLM_API_KEY          LLM API key, if the service needs one

Optional tuning, the library config keys of the same names:
HYPERMEM_LLM_{PATH,MAX_TOKENS,TIMEOUT,RETRIES} and
HYPERMEM_EMBEDDING_{BATCH_SIZE,PROTOCOL,TIMEOUT,RETRIES}.

search must use the same embedding settings the database was built with. add,
read and status need only HYPERMEM_DB.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
import functools
import inspect
import json
import os
import sys
from typing import Any

from hypermem.api import HyperMemory

_LLM_KEY_ENV = "HYPERMEM_LLM_API_KEY"
_EMBEDDING_KEY_ENV = "HYPERMEM_EMBEDDING_API_KEY"
_LLM_ENV = ("HYPERMEM_LLM_BASE_URL", "HYPERMEM_LLM_MODEL")
_EMBEDDING_ENV = ("HYPERMEM_EMBEDDING_URL", "HYPERMEM_EMBEDDING_MODEL", "HYPERMEM_EMBEDDING_DIMENSION")
# Optional settings: environment variable -> (config section, key, type).
_OPTIONAL_ENV = {
    "HYPERMEM_LLM_PATH": ("llm", "path", str),
    "HYPERMEM_LLM_MAX_TOKENS": ("llm", "max_tokens", int),
    "HYPERMEM_LLM_TIMEOUT": ("llm", "timeout", int),
    "HYPERMEM_LLM_RETRIES": ("llm", "retries", int),
    "HYPERMEM_EMBEDDING_BATCH_SIZE": ("embedding", "batch_size", int),
    "HYPERMEM_EMBEDDING_PROTOCOL": ("embedding", "protocol", str),
    "HYPERMEM_EMBEDDING_TIMEOUT": ("embedding", "timeout", int),
    "HYPERMEM_EMBEDDING_RETRIES": ("embedding", "retries", int),
}

_memory: HyperMemory | None = None
_build_job: Future | None = None
_build_progress: dict | None = None

# A SQLite connection may only be used on the thread that opened it, while the
# SDK runs sync tools on arbitrary worker threads. Every database call therefore
# runs on this single dedicated thread, which also keeps the event loop free.
# A background build occupies this thread until it finishes.
_DB_THREAD = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hypermem-db")


async def _on_db_thread(fn, *args, **kwargs):
    return await asyncio.get_running_loop().run_in_executor(_DB_THREAD, functools.partial(fn, *args, **kwargs))


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _int_env(name: str) -> int:
    try:
        return int(_env(name))
    except ValueError:
        raise RuntimeError(f"{name} must be an integer") from None


def _config_from_env() -> dict | None:
    """The library config, or None when the embedding settings are not configured."""
    if not all(_env(name) for name in _EMBEDDING_ENV):
        return None
    # Only build calls the LLM, so its section may stay empty for search alone.
    config = {"llm": {"base_url": _env("HYPERMEM_LLM_BASE_URL"), "model": _env("HYPERMEM_LLM_MODEL"),
                      "api_key_env": _LLM_KEY_ENV},
              "embedding": {"url": _env("HYPERMEM_EMBEDDING_URL"), "model": _env("HYPERMEM_EMBEDDING_MODEL"),
                            "dimension": _int_env("HYPERMEM_EMBEDDING_DIMENSION"),
                            "api_key_env": _EMBEDDING_KEY_ENV}}
    for name, (section, key, kind) in _OPTIONAL_ENV.items():
        if _env(name):
            config[section][key] = _int_env(name) if kind is int else _env(name)
    return config


def _require_services(tool: str, *, llm: bool = False) -> None:
    missing = [name for name in _EMBEDDING_ENV + (_LLM_ENV if llm else ()) if not _env(name)]
    if missing:
        raise RuntimeError(f"{tool} needs " + ", ".join(missing))


def _db_path() -> str:
    db = _env("HYPERMEM_DB")
    if not db:
        raise RuntimeError("Set HYPERMEM_DB to the path of an OS-Memory database")
    if db != ":memory:" and not os.path.isabs(db):
        raise RuntimeError(f"HYPERMEM_DB must be an absolute path: {db}")
    return db


def _get_memory(*, create: bool = False) -> HyperMemory:
    """Open the configured database once and reuse it for every tool call."""
    global _memory
    if _memory is None:
        db = _db_path()
        if db != ":memory:" and not os.path.exists(db):
            # Only add may create the file: opening a mistyped path from any
            # other tool would silently create an empty database.
            if not create:
                raise RuntimeError(f"HYPERMEM_DB does not exist: {db}; call add to create it")
            if not os.path.isdir(os.path.dirname(db)):
                raise RuntimeError(f"The directory of HYPERMEM_DB does not exist: {os.path.dirname(db)}")
        _memory = HyperMemory(db, config=_config_from_env())
    return _memory


def _build_running() -> bool:
    return _build_job is not None and not _build_job.done()


def _record_progress(info: dict) -> None:
    global _build_progress
    _build_progress = info


def _build_state() -> dict | None:
    if _build_job is None:
        return None
    if not _build_job.done():
        return {"state": "running", "progress": _build_progress}
    exc = _build_job.exception()
    if exc is not None:
        return {"state": "failed", "error": f"{type(exc).__name__}: {exc}", "progress": _build_progress}
    return {"state": "done", "report": _build_job.result()}


# -- tools ------------------------------------------------------------------

def add(records: list[dict[str, Any]] | None = None, jsonl_path: str | None = None,
        user: str | None = None) -> dict[str, int]:
    """Save records given inline or as a JSONL file; mirrors HyperMemory.add."""
    if (records is None) == (jsonl_path is None):
        raise ValueError("Pass exactly one of records (a list of records) or jsonl_path")
    memory = _get_memory(create=True)
    if records is not None:
        return memory.add(records, user=user)
    # The server runs in its own working directory, not the client's.
    if not os.path.isabs(jsonl_path):
        raise ValueError(f"jsonl_path must be an absolute path: {jsonl_path}")
    if not os.path.isfile(jsonl_path):
        raise ValueError(f"jsonl_path is not a file: {jsonl_path}")
    return memory._add_labelled(_jsonl_records(jsonl_path), user)


def _jsonl_records(path: str):
    """(label, record) per non-blank line, read lazily so a large file is streamed."""
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    record = json.loads(line)
                except ValueError as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON ({exc})") from None
                yield f"{path}:{number}", record


def build() -> dict[str, Any]:
    """Start HyperMemory.build in the background; status follows it."""
    global _build_job, _build_progress
    # Runs on the database thread, so a second build call queued behind this
    # one sees the job submitted here.
    if _build_running():
        raise RuntimeError("A build is already running; call status to follow it")
    _require_services("build", llm=True)
    memory = _get_memory()
    # The same checks build makes, here so that they fail the tool call itself
    # rather than the background job.
    if memory.is_built:
        raise RuntimeError("This memory is already built")
    observations = memory.count()["observations"]
    if observations == 0:
        raise RuntimeError("Add at least one memory with add before build")
    _build_progress = None
    _build_job = _DB_THREAD.submit(memory.build, progress=_record_progress)
    return {"status": "running", "observations": observations}


def search(query: str, top_k: int = 20, as_text: bool = True) -> dict[str, Any]:
    """Semantic search over the user's memory; mirrors HyperMemory.search."""
    _require_services("search")
    return _get_memory().search(query, top_k=top_k, as_text=as_text)


def read(event_id: str | None = None, start: str | None = None, end: str | None = None,
         entity: str | None = None, anchor: str | None = None, limit: int = 20,
         include_facts: bool = True, include_observations: bool = False,
         include_raw: bool = False, include_long_memories: bool = False) -> dict[str, Any]:
    """Structured browse of stored memories; mirrors HyperMemory.read."""
    return _get_memory().read(
        event_id=event_id, start=start, end=end, entity=entity, anchor=anchor, limit=limit,
        include_facts=include_facts, include_observations=include_observations,
        include_raw=include_raw, include_long_memories=include_long_memories)


def _database_state() -> dict[str, Any]:
    db = _db_path()
    services = {"search": _config_from_env() is not None,
                "build": _config_from_env() is not None and all(_env(name) for name in _LLM_ENV)}
    if _memory is None and db != ":memory:" and not os.path.exists(db):
        return {"db": db, "exists": False, "is_started": False, "is_built": False, "counts": None,
                "services": services}
    memory = _get_memory()
    return {"db": db, "exists": True, "is_started": memory.is_started, "is_built": memory.is_built,
            "counts": memory.count(), "services": services}


async def status() -> dict[str, Any]:
    """Database and build state; mirrors HyperMemory.is_started, is_built and count()."""
    job = _build_state()
    if job is not None and job["state"] == "running":
        # The database thread is busy building; report the job alone.
        return {"db": _db_path(), "build": job}
    state = await _on_db_thread(_database_state)
    state["build"] = job
    return state


_ADD_DESCRIPTION = (
    "Save memories. Pass exactly one of: records, a list of record objects; or jsonl_path, the absolute "
    "path on the server's machine of a UTF-8 JSONL file with one record per line (better for large "
    "batches). Every record has exactly {\"type\", \"timestamp\", \"data\"}; timestamp is when it happened on "
    "the device, integer Unix milliseconds or an ISO 8601 string. Everything else goes in data: optional "
    "from_app, record_id, anchors (order_id, transaction_id, tracking_id, reservation_id, phone, email, "
    "url), entities (contact, merchant, product, place, organizer), latitude/longitude (device position), "
    "plus by type: screen {action_description, typed_text?, title?, ocr?, image_path?}; gallery "
    "{image_description, ocr?, image_path?, captured_at?, location?}; sms {content, sender?, sender_name?}; "
    "push {action: receive|open, title and/or content}; calendar {title and/or content, start_time?, end_time?, remind_time?, calendar_type?}; "
    "clipboard {content, action?: copy|cut}; position {latitude+longitude and/or ssid, location?, "
    "accuracy?}; chat {query, answer?, session_id?}; recording {transcript: [{text, offset_ms, speaker "
    "(string or null)}], audio_path?}; note {action: create|edit|delete, title and/or content}. A bad "
    "record fails the whole batch, naming it. Returns the numbers of records and observations written. "
    "Records can be added only before build.")

_BUILD_DESCRIPTION = (
    "Construct the memory graph from everything added so far (DRED + SAGE). It calls the LLM and can take "
    "minutes, so it starts in the background and returns at once; call status to follow it until "
    "build.state is 'done' (with the build report) or 'failed'. If it failed because the LLM or embedding "
    "service failed, nothing is lost: once the service is back, call build again and it resumes. Search "
    "works only after a build is done.")

_SEARCH_DESCRIPTION = (
    "Search the user's long-term memory with a natural-language question (Chinese or English). "
    "Returns relevant events (title, summary, time span, grounded facts), long-term memories, "
    "and, unless as_text is false, a prompt-ready 'context' string. Use it for questions about what the "
    "user did, bought, planned, said or prefers.")

_READ_DESCRIPTION = (
    "Browse stored memories by structured filters instead of semantic search: an event_id, a time "
    "window (start/end as dates or times such as '2026-05-03' or '2026-05-03 08:30', local time unless an "
    "offset like '+08:00' is given; an end date alone includes that whole day), an entity name, or an "
    "anchor value such as an order ID. "
    "The include_* flags add each event's facts, observations, the original submitted records "
    "(include_raw), and related long-term memories. Returns events, long_memories and total.")

_STATUS_DESCRIPTION = (
    "Report the memory database and build state: whether the database exists, whether a build has started "
    "(no more add then) and finished, its counts, "
    "which tools the configured services allow, and the background build job (running with its latest "
    "progress, done with its report, or failed with the error).")


def _server_class():
    """The SDK's high-level server class and its ToolError: MCPServer in mcp 2.x, FastMCP in mcp 1.x."""
    try:
        from mcp.server.mcpserver import MCPServer
        from mcp.server.mcpserver.exceptions import ToolError
        return MCPServer, ToolError
    except ModuleNotFoundError as exc:
        if exc.name == "mcp":
            raise
        from mcp.server.fastmcp import FastMCP
        from mcp.server.fastmcp.exceptions import ToolError
        return FastMCP, ToolError


def build_server():
    """Create the MCP server with the add, build, search, read and status tools."""
    server_class, tool_error = _server_class()
    from mcp.types import ToolAnnotations
    # Read-only tools need no confirmation; add and build only ever add data.
    reading = ToolAnnotations(readOnlyHint=True)
    writing = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)

    def surfaced(fn):
        # Runs a sync tool on the dedicated database thread (refusing while a
        # build holds it), and turns configuration and usage errors (missing
        # env, unbuilt database, bad arguments) into ToolError: the SDK passes
        # only ToolError messages to the model, any other exception reaching it
        # as a bare "Error executing tool".
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            try:
                if inspect.iscoroutinefunction(fn):
                    return await fn(*args, **kwargs)
                if _build_running():
                    raise RuntimeError("A build is running; call status to follow it, then retry")
                return await _on_db_thread(fn, *args, **kwargs)
            except (RuntimeError, ValueError, OSError) as exc:
                raise tool_error(str(exc)) from exc
        return wrapper

    server = server_class("hypermemory")
    for fn, description, annotations in (
            (add, _ADD_DESCRIPTION, writing), (build, _BUILD_DESCRIPTION, writing),
            (search, _SEARCH_DESCRIPTION, reading), (read, _READ_DESCRIPTION, reading),
            (status, _STATUS_DESCRIPTION, reading)):
        server.tool(name=fn.__name__, description=description, annotations=annotations)(surfaced(fn))
    return server


def main() -> None:
    try:
        server = build_server()
    except ModuleNotFoundError as exc:
        if exc.name != "mcp":
            raise
        sys.stderr.write('The MCP server needs the optional dependency: pip install "hyperos-mem[mcp]"\n')
        raise SystemExit(1)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
