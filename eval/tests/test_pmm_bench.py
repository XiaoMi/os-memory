# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""Offline checks for the PMM-Bench entry point on the synthetic sample package."""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "examples" / "pmm_sample"
# The memory system's core package (hypermem/) is at the repository root.
HYPERMEM_SOURCE = ROOT.parent
sys.path[:0] = [str(ROOT), str(HYPERMEM_SOURCE)]

from pmm_bench import config as C  # noqa: E402
from pmm_bench import report as R  # noqa: E402
from pmm_bench.data import chinese_f1, load_events, load_questions, select_users  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class DataTests(unittest.TestCase):
    def test_users_and_questions(self):
        self.assertEqual(select_users(SAMPLE, "all"), ["U01", "U02"])
        self.assertEqual(select_users(SAMPLE, "U02"), ["U02"])
        with self.assertRaises(ValueError):
            select_users(SAMPLE, "U99")
        questions = load_questions(SAMPLE / "users" / "U01")
        self.assertEqual({q.category for q in questions}, {"Single", "Cross", "Temp", "Track", "Habit"})

    def test_both_inputs_load_in_time_order(self):
        for kind in ("cleaned", "raw"):
            events = load_events(SAMPLE / "users" / "U01", kind)
            self.assertEqual(len(events), 22)
            self.assertEqual([e.timestamp for e in events], sorted(e.timestamp for e in events))
        self.assertEqual(len(load_events(SAMPLE / "users" / "U01", "cleaned", limit=5)), 5)

    def test_f1(self):
        self.assertEqual(chinese_f1("京东 249 元", ["京东 249 元"]), 1.0)
        self.assertEqual(chinese_f1("", ["京东"]), 0.0)


class MiniTests(unittest.TestCase):
    def test_mini_keeps_questions_and_all_evidence(self):
        from pmm_bench.mini import make
        with tempfile.TemporaryDirectory() as tmp:
            stats = make(SAMPLE, "U01", Path(tmp), noise=2, seed=1, alias="demo")
            self.assertEqual(stats["evidence_lines"], stats["evidence_lines_in_cleaned"])
            self.assertEqual(stats["cleaned_noise_records"], 2)
            original = [q["question"] for q in json.loads((SAMPLE / "users" / "U01" / "qa.json").read_text())]
            self.assertEqual([q.question for q in load_questions(Path(tmp) / "users" / "demo")], original)
            self.assertEqual(len(load_events(Path(tmp) / "users" / "demo", "raw")), stats["raw_records"])


class ReportTests(unittest.TestCase):
    def test_judge_is_unweighted_mean_of_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            scores = {"Single": [1, 1, 1, 0], "Cross": [0], "Temp": [1], "Track": [0], "Habit": [1]}
            d = out / "results" / "demo" / "U01"
            d.mkdir(parents=True)
            n = 0
            for category, values in scores.items():
                for v in values:
                    n += 1
                    (d / f"{n}.json").write_text(json.dumps({"category": category, "status": "complete", "score": v, "f1": 0.5}))
            row = R.aggregate(out)["systems"]["demo"]
            self.assertAlmostEqual(row["judge"], (75 + 0 + 100 + 0 + 100) / 5)
            self.assertAlmostEqual(row["f1"], 50.0)


class EntryPointTests(unittest.TestCase):
    def run_cli(self, *args, env=None):
        return subprocess.run([sys.executable, "-m", "pmm_bench", *args], cwd=ROOT, env={**os.environ, **(env or {})},
                              capture_output=True, text=True, timeout=600)

    def test_adapter_path_two_users_resumable(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {"PMM_MOCK_LLM": "1", "PMM_PYTHON": sys.executable}
            first = self.run_cli("run", "--data", str(SAMPLE), "--systems", "mock,structmem", "--users", "all", "--out", tmp, env=env)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            self.assertIn("[structmem] U01: skipped", first.stdout)
            report = json.loads((Path(tmp) / "report.json").read_text())
            self.assertEqual(report["systems"]["mock"]["questions"], 13)
            second = self.run_cli("run", "--data", str(SAMPLE), "--systems", "mock", "--users", "U01", "--out", tmp, env=env)
            self.assertIn("[mock] U01: done (cached)", second.stdout)

    def test_protocol_requires_endpoints(self):
        cfg = C.load()
        with self.assertRaises(ValueError):
            C.hyperos_protocol({**cfg, "embedding": {**cfg["embedding"], "url_env": "PMM_TEST_UNSET"}})

    @unittest.skipUnless(__import__("importlib").util.find_spec("numpy"), "HyperOS-Mem needs numpy")
    def test_embedding_key_reaches_hyperos_settings(self):
        from pmm_bench.pmm_benchmark import settings_for
        env = {"OPENAI_BASE_URL": "http://llm/v1", "PMM_EMBEDDING_URL": "http://emb/v1/embeddings",
               "PMM_EMBEDDING_MODEL": "e"}
        with patch.dict(os.environ, env):
            protocol = C.hyperos_protocol(C.load())
        self.assertEqual(protocol["embedding"]["api_key_env"], "PMM_EMBEDDING_API_KEY")
        formation, sage = settings_for(protocol)
        self.assertEqual(formation["embedding"]["api_key_env"], "PMM_EMBEDDING_API_KEY")
        self.assertEqual(sage["embedding"]["api_key_env"], "PMM_EMBEDDING_API_KEY")

    @unittest.skipUnless(__import__("importlib").util.find_spec("numpy"), "HyperOS-Mem needs numpy")
    def test_hyperos_prefix_limit_and_changed_settings(self):
        port = free_port()
        server = subprocess.Popen([sys.executable, str(ROOT / "tests" / "mock_openai_server.py"), "--port", str(port), "--dimension", "64"])
        try:
            time.sleep(1)
            with tempfile.TemporaryDirectory() as tmp:
                base = f"http://127.0.0.1:{port}/v1"
                env = {"OPENAI_BASE_URL": base, "OPENAI_API_KEY": "offline", "PMM_EMBEDDING_URL": base + "/embeddings",
                       "PMM_EMBEDDING_MODEL": "offline", "NO_PROXY": "127.0.0.1", "no_proxy": "127.0.0.1"}
                run = Path(tmp) / "runs" / "hyperos-mem" / "U02" / "smoke"
                log = Path(tmp) / "logs" / "hyperos-mem" / "U02.log"

                def run_with(settings):
                    override = Path(tmp) / "override.json"
                    override.write_text(json.dumps(settings))
                    return self.run_cli("run", "--data", str(SAMPLE), "--systems", "hyperos-mem", "--users", "U02",
                                        "--out", tmp, "--config", str(override), "--limit", "7", env=env)

                first = run_with({"embedding": {"dimension": 64}})
                self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
                self.assertIn("[hyperos-mem] U02: ok\n", first.stdout)
                report = json.loads((run / "import_report.json").read_text())
                self.assertEqual((report["config"]["limit"], report["accepted_observations"]), (7, 7))

                second = run_with({"embedding": {"dimension": 64}, "models": {"construction": "another-model"}})
                self.assertIn("U02: ok (settings changed since the last run", second.stdout, second.stdout)
                self.assertEqual(json.loads((run / "config.json").read_text())["llm"]["model"], "another-model")
                revision = json.loads((run / "settings_revisions.json").read_text())[-1]
                self.assertIn({"setting": "llm.model", "from": "gpt-4.1-mini", "to": "another-model"}, revision["changes"])

                third = run_with({"embedding": {"dimension": 32}, "models": {"construction": "another-model"}})
                self.assertIn("U02: failed", third.stdout)
                self.assertIn("Embedding model or dimension changed (embedding.dimension: 64 -> 32)", log.read_text(encoding="utf-8"))
                self.assertEqual(json.loads((run / "config.json").read_text())["embedding"]["dimension"], 64)
        finally:
            server.terminate()
            server.wait()

    @unittest.skipUnless(__import__("importlib").util.find_spec("numpy"), "HyperOS-Mem needs numpy")
    def test_hyperos_final_path_with_offline_server(self):
        port = free_port()
        server = subprocess.Popen([sys.executable, str(ROOT / "tests" / "mock_openai_server.py"), "--port", str(port), "--dimension", "64"])
        try:
            time.sleep(1)
            with tempfile.TemporaryDirectory() as tmp:
                override = Path(tmp) / "override.json"
                override.write_text(json.dumps({"embedding": {"dimension": 64}}))
                base = f"http://127.0.0.1:{port}/v1"
                env = {"OPENAI_BASE_URL": base, "OPENAI_API_KEY": "offline", "PMM_EMBEDDING_URL": base + "/embeddings",
                       "PMM_EMBEDDING_MODEL": "offline", "NO_PROXY": "127.0.0.1", "no_proxy": "127.0.0.1"}
                result = self.run_cli("run", "--data", str(SAMPLE), "--systems", "hyperos-mem", "--users", "U02",
                                      "--out", tmp, "--config", str(override), env=env)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                summary = json.loads((Path(tmp) / "results" / "hyperos-mem" / "U02" / "summary.json").read_text())
                self.assertEqual(summary["overall"]["completed"], 5)
                self.assertTrue((Path(tmp) / "runs" / "hyperos-mem" / "U02" / "build" / "final_run_contract.json").exists())
        finally:
            server.terminate()
            server.wait()


if __name__ == "__main__":
    unittest.main()
