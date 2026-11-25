# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""An API failure stops the build without spending any task's retry budget, and
the same steps finish the memory once the service is restored."""
import copy
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import tempfile
import threading
import unittest

from test_idle_consolidation import model as wiki_model
from test_online_semantic import Encoder, observation, settings
from test_sage import AnyEncoder, sage_model
from test_sage_tools import model as tool_model
from hypermem import HyperMemory
from hypermem.memory import EventMemory
from hypermem.online.consolidation import records
from hypermem.online.engine import OnlineHyperMem as HyperMem
from hypermem.online.sage import config, rows
from hypermem.online.sage_tools import ToolSAGE
from hypermem.standardization import ModelServiceError, request_json


def down(*_):
    raise ModelServiceError("service down")


def tool_jobs(memory):
    return [json.loads(raw) for (raw,) in memory._connection.execute("SELECT record_json FROM sage_tool_jobs")]


class DownEncoder:
    def encode(self, texts):
        raise ModelServiceError("embedding service down")


class RequestClassificationTests(unittest.TestCase):
    def serve(self, status):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(status)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return {"base_url": f"http://127.0.0.1:{server.server_port}/v1", "path": "/chat/completions",
                "timeout": 5, "retries": 0}

    def test_service_failures_are_model_service_errors(self):
        for status in (401, 403, 404, 429, 500, 503):
            with self.subTest(status=status), self.assertRaises(ModelServiceError):
                request_json(self.serve(status), {"model": "m"})
        unreachable = {"base_url": "http://127.0.0.1:9/v1", "path": "/chat/completions", "timeout": 5, "retries": 0}
        with self.assertRaises(ModelServiceError):
            request_json(unreachable, {"model": "m"})

    def test_a_rejected_request_is_a_task_failure(self):
        for status in (400, 422):
            with self.subTest(status=status), self.assertRaises(RuntimeError) as caught:
                request_json(self.serve(status), {"model": "m"})
            self.assertNotIsInstance(caught.exception, ModelServiceError)


class ServiceFailureTests(unittest.TestCase):
    def engine(self, concurrency=1):
        memory = EventMemory()
        self.addCleanup(memory.close)
        vectors = {"a1": [1, 0, 0], "a2": [1, 0, 0], "b1": [0, 1, 0], "b2": [0, 1, 0]}
        for i, oid in enumerate(vectors):
            o = observation(oid, oid, i * 1200)
            memory.add_observation(o, {"input": o.to_dict()})
        engine = HyperMem(memory, settings(consolidation={"enabled": True, "concurrency": concurrency}),
                          encoder=Encoder(vectors), slow_resolver=wiki_model)
        engine.replay()
        return engine

    def tools(self, memory, concurrency=1, **kwargs):
        options = {"tool_resolver": tool_model, "resolver": sage_model, "encoder": AnyEncoder(), **kwargs}
        return ToolSAGE(memory, config({"embedding": {"dimension": 3}, "event": {"concurrency": concurrency}}),
                        **options)

    def test_boundary_consolidation_keeps_its_budget(self):
        for concurrency in (1, 2):
            with self.subTest(concurrency=concurrency):
                engine = self.engine(concurrency)
                engine.slow_resolver = down
                with self.assertRaises(ModelServiceError):
                    engine.run_slow(is_idle=True)
                tasks = records(engine.memory, "on_consolidation_tasks")
                self.assertTrue(tasks)
                self.assertEqual({t["attempts"] for t in tasks.values()}, {0})
                engine.slow_resolver = wiki_model
                self.assertEqual(engine.run_slow(is_idle=True)["status"], "complete")
                tasks = records(engine.memory, "on_consolidation_tasks")
                self.assertEqual({t["attempts"] for t in tasks.values() if t["status"] == "complete"}, {1})

    def test_tool_sage_keeps_its_budget(self):
        for concurrency in (1, 2):
            with self.subTest(concurrency=concurrency):
                engine = self.engine()
                engine.run_slow(is_idle=True)
                with self.assertRaises(ModelServiceError):
                    self.tools(engine.memory, concurrency, tool_resolver=down).run()
                jobs = tool_jobs(engine.memory)
                self.assertTrue(jobs)
                self.assertEqual({j["rounds"] for j in jobs}, {0})
                self.assertEqual(self.tools(engine.memory, concurrency).run()["status"], "complete")

    def test_embedding_failure_in_a_tool_resumes_from_the_saved_reply(self):
        reference = self.engine()
        reference.run_slow(is_idle=True)
        self.assertEqual(self.tools(reference.memory).run()["status"], "complete")
        engine = self.engine()
        engine.run_slow(is_idle=True)
        with self.assertRaises(ModelServiceError):
            self.tools(engine.memory, encoder=DownEncoder()).run()
        self.assertEqual(self.tools(engine.memory).run()["status"], "complete")
        rounds = lambda memory: sorted(j["rounds"] for j in tool_jobs(memory))
        self.assertEqual(rounds(engine.memory), rounds(reference.memory))

    def test_long_memory_keeps_its_budget(self):
        engine = self.engine()
        engine.run_slow(is_idle=True)
        with self.assertRaises(ModelServiceError):
            self.tools(engine.memory, resolver=down).run()
        tasks = rows(engine.memory, "sage_tasks")
        self.assertTrue(tasks)
        self.assertEqual({t["attempts"] for t in tasks.values()}, {0})
        self.assertEqual(self.tools(engine.memory).run()["status"], "complete")


class FakeServices(BaseHTTPRequestHandler):
    """The fake models above behind OpenAI-compatible endpoints; llm_up toggles the LLM."""
    llm_up = True

    def log_message(self, *_):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.endswith("/embeddings"):
            reply = {"data": [{"index": i, "embedding": [1.0, 0.1 * (i % 3), 0.0]} for i in range(len(body["input"]))]}
        elif not FakeServices.llm_up:
            self.send_response(503)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        elif "tools" in body:
            reply = {"choices": [{"message": tool_model(body["messages"], body["tools"], None)}]}
        else:
            payload = json.loads(body["messages"][1]["content"])
            try:
                content = sage_model(payload) if "response_format" in body else wiki_model(payload)
            except (KeyError, IndexError, TypeError):
                content = "invalid"  # a request shape the fakes do not cover: a bad reply, not an outage
            reply = {"choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}]}
        out = json.dumps(reply, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


class LibraryResumeTests(unittest.TestCase):
    def test_build_stops_on_an_outage_and_resumes(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeServices)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.addCleanup(setattr, FakeServices, "llm_up", True)
        base = f"http://127.0.0.1:{server.server_port}/v1"
        records = [{"type": "push", "timestamp": 1777770000000 + i * 60_000,
                    "data": {"from_app": "test", "action": "receive", "content": f"第{i}条通知"}} for i in range(4)]
        mem = HyperMemory(config={"llm": {"base_url": base, "model": "m", "retries": 0},
                                  "embedding": {"url": base + "/embeddings", "model": "e", "dimension": 3,
                                                "retries": 0}})
        self.addCleanup(mem.close)
        mem.add(records)
        FakeServices.llm_up = False
        with self.assertRaisesRegex(ModelServiceError, r"call build\(\) again to continue"):
            mem.build()
        self.assertEqual((mem.is_started, mem.is_built), (True, False))
        with self.assertRaisesRegex(RuntimeError, "already built or being built"):
            mem.add(records)
        with self.assertRaisesRegex(RuntimeError, "the last build did not finish"):
            mem.search("通知")
        FakeServices.llm_up = True
        report = mem.build()
        self.assertEqual((report["status"], report["resumed"]), ("built", True))
        self.assertTrue(mem.is_built)
        self.assertIn("events", mem.search("通知"))

    def test_a_resumed_build_may_use_new_service_settings_but_not_a_new_embedding(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeServices)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            nothing_listens = s.getsockname()[1]
        good = f"http://127.0.0.1:{server.server_port}/v1"
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = str(Path(tmp.name) / "memory.db")

        def open_with(llm_base, llm_model="m", dimension=3):
            return HyperMemory(path, config={"llm": {"base_url": llm_base, "model": llm_model, "retries": 0},
                                             "embedding": {"url": good + "/embeddings", "model": "e",
                                                           "dimension": dimension, "retries": 0}})

        with open_with(f"http://127.0.0.1:{nothing_listens}/v1") as mem:
            mem.add([{"type": "push", "timestamp": 1777770000000 + i * 60_000,
                      "data": {"from_app": "test", "action": "receive", "content": f"第{i}条通知"}} for i in range(4)])
            with self.assertRaises(ModelServiceError):
                mem.build()
            self.assertEqual((mem.is_started, mem.is_built), (True, False))
        with open_with(good, dimension=4) as mem, self.assertRaisesRegex(
                ValueError, r"Embedding model or dimension changed \(embedding.dimension: 3 -> 4\)"):
            mem.build()
        with open_with(good, llm_model="m2") as mem:
            report = mem.build()
            self.assertEqual((report["status"], report["resumed"]), ("built", True))


class ResumableSettingsTests(unittest.TestCase):
    def test_only_the_service_settings_may_change(self):
        from hypermem.config import check_resumable, config as formation_config
        base = formation_config({"embedding": {"dimension": 3}})
        check_resumable(None, base)
        services = copy.deepcopy(base)
        services["llm"].update(base_url="http://other/v1", model="m2", api_key_env="K", timeout=5)
        services["embedding"].update(base_url="http://emb", api_key_env="K", batch_size=8, retries=5)
        check_resumable(base, services)
        for key, value in (("model", "other"), ("dimension", 4), ("backend", "http")):
            changed = copy.deepcopy(base)
            changed["embedding"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, rf"Embedding model or dimension changed \(embedding.{key}"):
                check_resumable(base, changed)
        method = copy.deepcopy(base)
        method["online"]["tau_s"] = 0.5
        with self.assertRaisesRegex(ValueError, "Different settings require a new memory run"):
            check_resumable(base, method)


if __name__ == "__main__":
    unittest.main()
