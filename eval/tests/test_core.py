# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hypereval.adapters.amem import AMEMAdapter
from hypereval.adapters.mem0 import Mem0Adapter
from hypereval.adapters.simplemem import SimpleMemAdapter
from hypereval.runner import run_benchmark
from hypereval.registry import _manifest, check_system
from hypereval.datasets import LoCoMoDataset
from hypereval.datasets import LongMemEvalDataset
from hypereval.metrics import bleu1


class CoreTests(unittest.TestCase):
    def test_toy_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = run_benchmark("mock", "toy", output_dir=directory)
            metrics = json.loads((Path(root) / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["questions"], 1)
            self.assertEqual(metrics["completed_samples"], 1)

    def test_lightmem_manifest_is_wired(self):
        manifest, path = _manifest("lightmem")
        self.assertIsNotNone(path)
        self.assertEqual(manifest["status"], "ready")
        self.assertEqual(manifest["adapter"], "hypereval.adapters.lightmem:LightMemAdapter")

    def test_amem_manifest_is_wired(self):
        manifest, path = _manifest("amem")
        self.assertIsNotNone(path)
        self.assertEqual(manifest["status"], "ready")
        self.assertEqual(manifest["adapter"], "hypereval.adapters.amem:AMEMAdapter")

    def test_mem0_manifest_is_wired(self):
        manifest, path = _manifest("mem0")
        self.assertIsNotNone(path)
        self.assertEqual(manifest["status"], "ready")
        self.assertEqual(manifest["adapter"], "hypereval.adapters.mem0:Mem0Adapter")

    def test_simplemem_manifest_is_wired(self):
        manifest, path = _manifest("simplemem")
        self.assertIsNotNone(path)
        self.assertEqual(manifest["status"], "ready")
        self.assertEqual(manifest["backend"], "text + Omni-SimpleMem multimodal")
        self.assertEqual(manifest["adapter"], "hypereval.adapters.simplemem:SimpleMemAdapter")

    def test_model_names_are_configurable_and_keys_are_read_from_environment(self):
        config = {
            "model": {
                "provider": "openai",
                "name": "test-model",
                "api_key_env": "OPENAI_API_KEY",
            }
        }
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            amem = AMEMAdapter()
            amem._config = config
            built_amem = amem._build_amem_config()
            self.assertEqual(built_amem["llm_model"], "test-model")
            self.assertEqual(built_amem["api_key"], "test-key")

            mem0 = Mem0Adapter()
            mem0._config = config
            built_mem0 = mem0._build_mem0_config()
            self.assertEqual(built_mem0["llm"]["config"]["model"], "test-model")
            self.assertEqual(built_mem0["llm"]["config"]["api_key"], "test-key")

            simplemem = SimpleMemAdapter()
            simplemem._config = config
            built_simplemem = simplemem._build_simplemem_config()
            self.assertEqual(built_simplemem["model"], "test-model")
            self.assertEqual(built_simplemem["api_key"], "test-key")

    def test_locomo_loader_normalizes_common_shape(self):
        payload = [{
            "sample_id": "conv-1",
            "session_1_date_time": "2026-07-01T10:00:00Z",
            "session_1": [{"speaker": "user", "text": "I like tea."}],
            "qa": [{"question": "What do I like?", "answer": "tea", "evidence": ["I like tea."], "category": 1}],
        }]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "locomo.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            samples = LoCoMoDataset().load(str(path))
        self.assertEqual(len(samples), 1)
        self.assertEqual(len(samples[0].events), 1)
        self.assertEqual(samples[0].questions[0].answers, ("tea",))
        self.assertEqual(samples[0].questions[0].evidence_event_ids, ("conv-1-session_1-e1",))

    def test_locomo_loader_preserves_category_five_contract(self):
        payload = [{
            "sample_id": "conv-1",
            "session_1": [{"speaker": "user", "text": "I like tea."}],
            "qa": [{
                "question": "What coffee do I like?",
                "adversarial_answer": "espresso",
                "category": 5,
            }],
        }]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "locomo.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            question = LoCoMoDataset().load(str(path))[0].questions[0]
        self.assertEqual(question.answers, ())
        self.assertTrue(question.metadata["unanswerable"])
        self.assertEqual(question.metadata["adversarial_answer"], "espresso")

    def test_longmemeval_loader_builds_one_isolated_episode(self):
        payload = [{
            "question_id": "q-1",
            "question_type": "single-session-user",
            "question": "What color do I like?",
            "answer": "blue",
            "question_date": "2024-01-03",
            "haystack_session_ids": ["s1"],
            "haystack_dates": ["2024-01-01"],
            "haystack_sessions": [[{
                "role": "user", "content": "I like blue.", "has_answer": True,
            }]],
            "answer_session_ids": ["s1"],
        }]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "longmemeval.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            samples = LongMemEvalDataset().load(str(path), sample_size=2)
        self.assertEqual(len(samples), 1)
        self.assertEqual(len(samples[0].events), 1)
        self.assertEqual(samples[0].questions[0].answers, ("blue",))
        self.assertEqual(samples[0].questions[0].evidence_event_ids, ("q-1-s1-e1",))

    def test_bleu1_exact_and_brevity_penalty(self):
        self.assertEqual(bleu1("the cat sat", ["the cat sat"]), 1.0)
        self.assertLess(bleu1("the cat", ["the cat sat"]), 1.0)


if __name__ == "__main__":
    unittest.main()
