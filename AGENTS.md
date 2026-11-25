# AGENTS.md — OS-Memory

This file tells coding agents (Claude Code, Codex, Cursor, …) how to install and connect OS-Memory. If you are here to change the code rather than install it, read [Changing the code](#changing-the-code) instead.

OS-Memory is a Python library (package `hyperos-mem`, import name `hypermem`) that builds a personal memory graph from records (`type`, `timestamp` and a `data` object) and answers questions over it. It can be used two ways:

- **Library (pip)** — `from hypermem import HyperMemory`; save memories with `add`, build the graph with `build`, then `search` and `read` them. Full guide: [`README.md`](README.md) (Chinese: [`README.zh.md`](README.zh.md)).
- **MCP server** — exposes the library methods as MCP tools of the same names, `add`, `build`, `search` and `read`, plus `status`, to an MCP client (Claude Desktop, Cursor, Claude Code, …). The database can be built through these tools or offline with the library.

If the user did not say which one they want, install the library, then ask whether they also want the MCP server.

## Before you start

1. Check Python: `python --version` must print 3.10 or newer. On some systems the command is `python3`; use whichever works in every command below.
2. Check git: `git --version`.
3. Check the operating system. Linux and macOS are supported. **On Windows, file-backed memory databases cannot be opened** (the writer lock uses POSIX `fcntl`), so a database cannot be built or served there. Tell the user and offer to install inside WSL instead; do not try to work around it.
4. Ask the user for the model services (the library needs them to build and query memory; nothing works without them):
   - an OpenAI-compatible **LLM** base URL ending in `/v1`, and a model name — used by `build`;
   - an OpenAI-compatible **embedding** URL ending in `/v1/embeddings`, a model name, and its vector dimension — used by `build` and `search`;
   - the API key, if the services need one. Keep keys in environment variables or the MCP client config. **Never write a key into any file in the repository.**

## 1. Get the code

Skip this step if the repository is already cloned and the current directory is its root (it contains `AGENTS.md`, `pyproject.toml` and `hypermem/`).

```bash
git clone https://github.com/XiaoMi/os-memory.git
cd os-memory
```

Use the repository URL the user gave you. If they gave none, ask for it.

## 2. Create a virtual environment

Install into a dedicated environment, not the system Python.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows (WSL not used): .venv\Scripts\activate
```

## 3. Install

Library only:

```bash
pip install -e .
```

Library and MCP server (adds the `mcp` SDK):

```bash
pip install -e ".[mcp]"
```

## 4. Verify

```bash
python -c "from hypermem import HyperMemory; print('hypermem ok')"
```

Expected output: `hypermem ok`. If you installed the MCP extra, also run:

```bash
python -c "from hypermem.mcp_server import build_server; build_server(); print('mcp ok')"
```

Expected output: `mcp ok`. `ModuleNotFoundError: No module named 'mcp'` means the MCP extra is missing; rerun the second install command.

The library install is done here. Point the user to `README.md` for the record format (chapter 4) and API (chapter 5); `README.zh.md` carries the same content in Chinese. Continue only if they want the MCP server.

## 5. MCP: choose the memory database

`HYPERMEM_DB` is the absolute path of the memory database the server works on. Pick one of:

- **The user already has a built database**: use its path.
- **Build through the MCP tools** (step 7): choose a new absolute path whose directory exists. The first `add` creates the file.
- **Build offline with the library**: run the script below, then use its database path. Use this for large inputs or when the client should never call the LLM.

Every record has exactly three top-level fields: `type`, `timestamp` (integer Unix milliseconds or an ISO 8601 string) and `data`. Everything else goes inside `data`: the optional common fields `from_app`, `record_id`, `anchors`, `entities`, `latitude`/`longitude` (where the device was), plus the fields of that type (the 10 record types and their fields are in `README.md`, chapter 4). The user's data may be in any format; convert it to these records, and ask the user where it is and what it means rather than guessing. Do not invent data. Records already in a JSONL file (one per line) can be read as below. To build offline, run, with the values from "Before you start" filled in:

```bash
export HYPERMEM_LLM_API_KEY=...         # only if the LLM service needs a key
export HYPERMEM_EMBEDDING_API_KEY=...   # only if the embedding service needs a key
python - <<'EOF'
import json
from hypermem import HyperMemory

config = {
    "llm": {"base_url": "<LLM base URL ending in /v1>", "model": "<LLM model>",
            "api_key_env": "HYPERMEM_LLM_API_KEY"},
    "embedding": {"url": "<embedding URL ending in /v1/embeddings>", "model": "<embedding model>",
                  "dimension": <dimension>, "api_key_env": "HYPERMEM_EMBEDDING_API_KEY"},
}
with open("/absolute/path/to/records.jsonl", encoding="utf-8") as f, \
        HyperMemory("/absolute/path/to/memory.db", config=config) as mem:
    print(mem.add(json.loads(line) for line in f if line.strip()))
    print(mem.build())
EOF
```

`build` calls the LLM and can take minutes for large inputs. It is finished when it prints a dict with `'status': 'built'`.

If it raises `ModelServiceError`, the LLM or embedding service failed (unreachable, timing out, overloaded, or refusing the key, URL or model; the message names the URL). Nothing is lost: ask the user to fix the service, then run only the `build()` part again on the same database (not `add`: the records are already stored). It resumes where it stopped. The rerun may use a corrected LLM or embedding URL, key, or another LLM model; it must keep the embedding model and dimension, or `build` stops with `Embedding model or dimension changed …` and the records must go into a new database.

## 6. MCP: register the server with the client

Use **absolute paths** everywhere: MCP clients start the server as a subprocess and do not see the activated virtual environment or the current directory. Get the interpreter path with `python -c "import sys; print(sys.executable)"` while the virtual environment is active.

The embedding settings must be the same ones used to build the database. `add`, `read` and `status` need only `HYPERMEM_DB`; `search` also needs the three `HYPERMEM_EMBEDDING_*` settings; `build` needs those and the two `HYPERMEM_LLM_*` settings. Leave the `HYPERMEM_LLM_*` settings out if the database was built offline.

Server configuration (the same JSON for Claude Desktop and Cursor):

```json
{
  "mcpServers": {
    "hypermemory": {
      "command": "/absolute/path/to/os-memory/.venv/bin/python",
      "args": ["-m", "hypermem.mcp_server"],
      "env": {
        "HYPERMEM_DB": "/absolute/path/to/memory.db",
        "HYPERMEM_EMBEDDING_URL": "<embedding URL ending in /v1/embeddings>",
        "HYPERMEM_EMBEDDING_MODEL": "<embedding model>",
        "HYPERMEM_EMBEDDING_DIMENSION": "<dimension>",
        "HYPERMEM_EMBEDDING_API_KEY": "<key, only if needed>",
        "HYPERMEM_LLM_BASE_URL": "<LLM base URL ending in /v1>",
        "HYPERMEM_LLM_MODEL": "<LLM model>",
        "HYPERMEM_LLM_API_KEY": "<key, only if needed>"
      }
    }
  }
}
```

Where it goes:

| Client | Location |
| --- | --- |
| Claude Desktop (macOS) | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Claude Desktop (Windows) | `%APPDATA%\Claude\claude_desktop_config.json` |
| Cursor | `~/.cursor/mcp.json` (all projects) or `.cursor/mcp.json` (one project) |

If the file already exists, add the `hypermemory` entry inside its existing `mcpServers` object; do not overwrite other servers. Ask the user before writing the API key into the file, or let them paste it themselves.

Claude Code registers the server with a command instead of a file:

```bash
claude mcp add hypermemory \
  -e HYPERMEM_DB=/absolute/path/to/memory.db \
  -e HYPERMEM_EMBEDDING_URL=<embedding URL> \
  -e HYPERMEM_EMBEDDING_MODEL=<embedding model> \
  -e HYPERMEM_EMBEDDING_DIMENSION=<dimension> \
  -e HYPERMEM_LLM_BASE_URL=<LLM base URL> \
  -e HYPERMEM_LLM_MODEL=<LLM model> \
  -- /absolute/path/to/os-memory/.venv/bin/python -m hypermem.mcp_server
```

Optional tuning settings, matching the library config keys of the same names, are listed in `README.md` (section 2.3, "Option 2: install the MCP server"; `README.zh.md` is the Chinese version).

Restart Claude Desktop or Cursor after editing the config. The client should then list the tools `add`, `build`, `search`, `read` and `status`. Ask it to call `status`: it reports the database path, whether it exists and is built, and which tools the configured services allow.

## 7. MCP: build through the client (optional)

Skip this step if the database was built offline. Otherwise, in the client:

1. Call `add` with exactly one of: `records`, a list of records (for a small batch); or `jsonl_path`, the **absolute** path of a JSONL file with one record per line (for a large batch). It returns the numbers of records and observations written; a bad record fails the whole batch and names it (`records[N]` or `file:line`).
2. Call `build`. It returns `{"status": "running", ...}` at once and builds in the background.
3. Call `status` until `build.state` is `done` (its `report` has `'status': 'built'`) or `failed` (its `error` says why). This can take minutes for large inputs. While the build runs, the other tools return `A build is running`.
4. If `build.state` is `failed` and the error starts with `ModelServiceError`, the LLM or embedding service failed. Ask the user to fix the service (URL, key, model or quota; restart the client if they changed its `env`), then call `build` again: it resumes where it stopped and loses no work.

Then ask something like "我最近订了哪些火车票？" to confirm `search` answers from the database.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| Tool error `Set HYPERMEM_DB to the path of an OS-Memory database` | `HYPERMEM_DB` is missing from the server's `env`. |
| Tool error `HYPERMEM_DB must be an absolute path: …` | Replace the relative path with an absolute one. |
| Tool error `HYPERMEM_DB does not exist: …; call add to create it` | Wrong path, or the database is new and nothing was added yet. Fix the path, or call `add` first. |
| Tool error `search needs …` / `build needs …` | The listed settings are missing from `env`. |
| Tool error `jsonl_path must be an absolute path` / `jsonl_path is not a file` | Pass the absolute path of an existing JSONL file on the machine the server runs on. |
| Tool error `records[N]: …` or `<file>:<line>: …` | That record is invalid (missing `data` field, a top-level field other than `type`/`timestamp`/`data`, bad `timestamp`, bad `action` value, duplicate `record_id`). Fix it and call `add` again; nothing from the batch was written. |
| Tool error `Call build() before search()…` | The database was never built, or its last build did not finish. Build it (step 5 or 7); an unfinished build resumes. |
| Tool error `This memory is already built` / `already built or being built` | A database is built once, and records can only be added before its build starts. Use a new `HYPERMEM_DB` for further records. |
| `build.state` is `failed` with `ModelServiceError: …` | The LLM or embedding service failed. Fix the service, then call `build` again; it resumes. |
| `build.state` is `failed` with `Embedding model or dimension changed …` | The embedding model or dimension differs from the one this database was built with. Restore it, or use a new `HYPERMEM_DB` and add the records again. |
| Tool error `A build is running…` | Wait: call `status` until `build.state` is `done` or `failed`. |
| `ModuleNotFoundError: No module named 'fcntl'` | Windows. Use WSL (see "Before you start"). |
| The client shows no `hypermemory` tools | The `command` path is wrong or relative, or the client was not restarted. Run the `command` with its `args` in a terminal: it should start and wait silently for input (stop it with Ctrl+C). |

## Changing the code

The method is DRED (fast path) + SAGE (slow path: boundary consolidation, full ToolSAGE, LongMemory), and `hypermem/` contains only this implementation. Do not add alternative formation modes, SAGE variants or switches that skip a step. The supported library interface is `hypermem.HyperMemory` (`add` / `build` / `search` / `read`; see `README.md`), and `hypermem.mcp_server` exposes it to MCP clients as tools of the same names, `add` / `build` / `search` / `read`, plus `status` (`build` runs in the background and `status` follows it). Keep the MCP tools' parameters identical to the library methods'. The PMM-Bench harness, in the separate `eval/` tree, drives the same pipeline through `pmm_bench.final`.

Read `docs/VERSION.md` and `docs/final_version.json` before describing, running or changing OS-Memory; read `docs/SAGE_TOOLS.md` when working on SAGE. Run the core tests from the repository root with `python -m unittest discover -s tests -t tests`.

Evaluation on PMM-Bench goes through `python -m pmm_bench` from the `eval/` tree (see `eval/docs/EVALUATION.md`). Every system, OS-Memory included, shares `pmm_bench/prompts.py` and the scoring code.

Never commit credentials, endpoint addresses of private services, real user data, run databases or evaluation outputs.
