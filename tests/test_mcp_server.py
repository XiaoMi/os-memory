# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

import asyncio
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from hypermem import ModelServiceError, api, mcp_server

try:
    import mcp  # noqa: F401
except ModuleNotFoundError:
    mcp = None

RECORDS = [
    {"type": "screen", "timestamp": 1777770000000,
     "data": {"from_app": "铁路12306", "action_description": "查询 5月8日 北京南→上海虹桥 G15 二等座"}},
    {"type": "sms", "timestamp": "2026-05-03T09:06:00+08:00",
     "data": {"from_app": "短信", "content": "订单E2026050301已支付", "sender": "12306",
              "anchors": {"order_id": "E2026050301"}}},
]
EMBEDDING = {"HYPERMEM_EMBEDDING_URL": "http://127.0.0.1:9/v1/embeddings",
             "HYPERMEM_EMBEDDING_MODEL": "test", "HYPERMEM_EMBEDDING_DIMENSION": "8"}
LLM = {"HYPERMEM_LLM_BASE_URL": "http://127.0.0.1:9/v1", "HYPERMEM_LLM_MODEL": "test"}


class ProgressCallbackTests(unittest.TestCase):
    def test_every_stage_reports_one_dict(self):
        seen = []
        dred, boundary, tools = api._progress_callbacks(seen.append)
        dred(3, {"events": 2})
        boundary(phase="split", step="prepare")
        tools({"stage": "sage_event_tools", "pass": 1})
        self.assertEqual(seen, [{"stage": "dred", "events": 2, "processed": 3},
                                {"stage": "sage_boundary", "phase": "split", "step": "prepare"},
                                {"stage": "sage_event_tools", "pass": 1}])
        self.assertEqual(api._progress_callbacks(None), (None, None, None))


@unittest.skipIf(mcp is None, "needs the mcp extra")
class McpServerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.db = str(self.root / "memory.db")
        self.records = str(self.root / "records.jsonl")
        Path(self.records).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in RECORDS) + "\n",
                                      encoding="utf-8")
        environ = patch.dict(os.environ, {"HYPERMEM_DB": self.db}, clear=False)
        environ.start()
        self.addCleanup(environ.stop)
        for name in list(EMBEDDING) + list(LLM):
            os.environ.pop(name, None)
        self.addCleanup(self.reset)
        self.server = mcp_server.build_server()
        self.tool_error = mcp_server._server_class()[1]

    def reset(self):
        if mcp_server._build_job is not None:
            mcp_server._build_job.exception(timeout=10)  # wait, without raising
        if mcp_server._memory is not None:
            mcp_server._DB_THREAD.submit(mcp_server._memory.close).result()
        mcp_server._memory = mcp_server._build_job = mcp_server._build_progress = None

    def call(self, name, **arguments):
        result = asyncio.run(self.server.call_tool(name, arguments))
        # mcp 2.x returns a CallToolResult, mcp 1.x a (content, structured) pair.
        structured = result.structured_content if hasattr(result, "structured_content") else result[1]
        return structured["result"] if set(structured) == {"result"} else structured

    def assertToolError(self, message, name, **arguments):
        with self.assertRaises(self.tool_error) as caught:
            self.call(name, **arguments)
        self.assertIn(message, str(caught.exception))

    def test_tools_mirror_the_library(self):
        tools = {tool.name: tool for tool in asyncio.run(self.server.list_tools())}
        self.assertEqual(set(tools), {"add", "build", "search", "read", "status"})
        # mcp 2.x names the fields in snake_case, mcp 1.x in camelCase.
        field = lambda item, snake, camel: getattr(item, snake) if hasattr(item, snake) else getattr(item, camel)
        schema = lambda name: set(field(tools[name], "input_schema", "inputSchema")["properties"])
        read_only = lambda name: field(tools[name].annotations, "read_only_hint", "readOnlyHint")
        self.assertEqual(schema("add"), {"records", "jsonl_path", "user"})
        self.assertEqual(schema("search"), {"query", "top_k", "as_text"})
        self.assertEqual(schema("build"), set())
        self.assertTrue(read_only("read"))
        self.assertTrue(read_only("status"))
        self.assertFalse(read_only("add"))
        self.assertFalse(read_only("build"))

    def test_missing_database_is_created_only_by_add(self):
        state = self.call("status")
        self.assertEqual((state["exists"], state["is_started"], state["is_built"], state["build"]),
                         (False, False, False, None))
        self.assertToolError("call add to create it", "read")
        self.assertFalse(os.path.exists(self.db))
        self.assertEqual(self.call("add", records=RECORDS, user="U01"), {"records": 2, "observations": 2})
        state = self.call("status")
        self.assertTrue(state["exists"])
        self.assertEqual((state["is_started"], state["is_built"]), (False, False))
        self.assertEqual(state["counts"]["observations"], 2)
        self.assertEqual(self.call("add", jsonl_path=self.records), {"records": 2, "observations": 2})
        self.assertEqual(state["services"], {"search": False, "build": False})
        self.assertEqual(self.call("read")["total"], 0)  # no events before build
        self.assertEqual(self.call("read", start="2026-05-03", end="2026-05-03 18:00")["total"], 0)
        self.assertToolError("start must be a date or time", "read", start="yesterday")

    def test_embedding_settings_alone_serve_queries(self):
        # README 2.3: the LLM settings may be omitted when the server only queries.
        with patch.dict(os.environ, EMBEDDING):
            self.assertEqual(self.call("add", records=RECORDS, user="U01"), {"records": 2, "observations": 2})
            state = self.call("status")
            self.assertEqual(state["services"], {"search": True, "build": False})
            self.assertEqual(state["counts"]["observations"], 2)
            self.assertEqual(self.call("read")["total"], 0)
            self.assertToolError("Call build() before search()", "search", query="订单")
            self.assertToolError("build needs HYPERMEM_LLM_BASE_URL, HYPERMEM_LLM_MODEL", "build")

    def test_usage_errors_reach_the_client(self):
        self.assertToolError("must be an absolute path", "add", jsonl_path="records.jsonl")
        self.assertToolError("is not a file", "add", jsonl_path=str(self.root / "missing.jsonl"))
        self.assertToolError("exactly one of records", "add")
        self.assertToolError("exactly one of records", "add", records=RECORDS, jsonl_path=self.records)
        self.assertToolError("records[1]: type 'sms' requires data.content", "add",
                             records=[RECORDS[0], {**RECORDS[1], "data": {"sender": "12306"}}])
        Path(self.records).write_text('\n{"type": "sms", "timestamp": 1777770000000, "data": {}}\n', encoding="utf-8")
        self.assertToolError("records.jsonl:2: type 'sms' requires data.content", "add", jsonl_path=self.records)
        Path(self.records).write_text('{"type": \n', encoding="utf-8")
        self.assertToolError("records.jsonl:1: invalid JSON", "add", jsonl_path=self.records)
        self.assertToolError("search needs HYPERMEM_EMBEDDING_URL", "search", query="订单")
        with patch.dict(os.environ, EMBEDDING):
            self.assertToolError("build needs HYPERMEM_LLM_BASE_URL, HYPERMEM_LLM_MODEL", "build")
        with patch.dict(os.environ, {"HYPERMEM_DB": "memory.db"}):
            self.assertToolError("HYPERMEM_DB must be an absolute path", "status")

    def test_build_runs_in_the_background(self):
        self.call("add", jsonl_path=self.records)
        release = threading.Event()

        def fake_build(memory, *, progress=None):
            progress({"stage": "dred", "processed": 1})
            release.wait(10)
            return {"status": "built", "observations": 2}

        with patch.dict(os.environ, {**EMBEDDING, **LLM}), patch.object(api.HyperMemory, "build", fake_build):
            self.assertEqual(self.call("build"), {"status": "running", "observations": 2})
            self.assertEqual(self.call("status")["build"]["state"], "running")
            self.assertToolError("A build is running", "read")
            self.assertToolError("A build is running", "build")
            release.set()
            mcp_server._build_job.result(timeout=10)
            state = self.call("status")
        self.assertEqual(state["build"], {"state": "done", "report": {"status": "built", "observations": 2}})
        self.assertEqual(state["services"], {"search": True, "build": True})

    def test_failed_build_is_reported_and_can_be_resumed(self):
        self.call("add", jsonl_path=self.records)

        def fake_build(memory, *, progress=None):
            raise ModelServiceError("Model service down. call build() again to continue.")

        with patch.dict(os.environ, {**EMBEDDING, **LLM}), patch.object(api.HyperMemory, "build", fake_build):
            self.call("build")
            mcp_server._build_job.exception(timeout=10)
            job = self.call("status")["build"]
            self.assertEqual(job["state"], "failed")
            self.assertEqual(job["error"], "ModelServiceError: Model service down. call build() again to continue.")
            # Not built, so build is accepted again and resumes.
            self.assertEqual(self.call("build")["status"], "running")
            mcp_server._build_job.exception(timeout=10)

    def test_optional_settings_reach_the_library_config(self):
        with patch.dict(os.environ, {**EMBEDDING, **LLM, "HYPERMEM_LLM_TIMEOUT": "30",
                                     "HYPERMEM_EMBEDDING_BATCH_SIZE": "4", "HYPERMEM_EMBEDDING_PROTOCOL": "openai"}):
            config = mcp_server._config_from_env()
        self.assertEqual(config["llm"]["timeout"], 30)
        self.assertEqual(config["embedding"]["batch_size"], 4)
        self.assertEqual(config["embedding"]["protocol"], "openai")
        with patch.dict(os.environ, {**EMBEDDING, "HYPERMEM_LLM_RETRIES": "two"}):
            with self.assertRaisesRegex(RuntimeError, "HYPERMEM_LLM_RETRIES must be an integer"):
                mcp_server._config_from_env()


if __name__ == "__main__":
    unittest.main()
