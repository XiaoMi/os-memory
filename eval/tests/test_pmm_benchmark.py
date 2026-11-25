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
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

# pmm_benchmark now lives in the eval tree (pmm_bench); the memory system's own
# modules are in hypermem/ at the repository root, and raw_record comes from the
# core test suite in the root tests/.
ROOT = Path(__file__).resolve().parents[1]
HYPERMEM_SOURCE = ROOT.parent
sys.path[:0] = [str(ROOT), str(HYPERMEM_SOURCE), str(HYPERMEM_SOURCE / "tests")]

from hypermem.import_observations import import_jsonl  # noqa: E402
from hypermem.memory import EventMemory  # noqa: E402
from pmm_bench.pmm_benchmark import answer_context, parse_judge, summarize  # noqa: E402
from pmm_bench.pmm_benchmark import evaluate, verify_run_code, file_hash  # noqa: E402
from hypermem.source_verify import verify_source  # noqa: E402
from hypermem.online.io import write_json  # noqa: E402
from test_import_observations import raw_record  # noqa: E402


class PMMBenchmarkTests(unittest.TestCase):

    def test_explicit_sage_retry_is_forwarded(self):
        from pmm_bench.pmm_benchmark import main
        with tempfile.TemporaryDirectory() as tmp:
            args = ["pmm", "--package", tmp, "--output", tmp,
                    "--stage", "sage", "--retry-unresolved"]
            with patch("sys.argv", args), patch("pmm_bench.pmm_benchmark.sage") as sage:
                main()
                sage.assert_called_once_with(Path(tmp)/"build",retry_unresolved=True)

    def test_retry_rejects_other_stages_before_reading_credentials_or_writing(self):
        from pmm_bench.pmm_benchmark import main
        for stage in ("prepare","dred","evaluate","dashboard"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                args = ["pmm", "--package", tmp, "--output", str(root/"unused"),
                        "--stage", stage, "--retry-unresolved", "--env-file", str(root/"missing.env")]
                with patch("sys.argv", args), patch("sys.stderr"), self.assertRaises(SystemExit) as exc:
                    main()
                self.assertEqual(exc.exception.code,2)
                self.assertFalse((root/"unused").exists())

    def test_dashboard_title_is_run_specific_and_escaped(self):
        from hypermem.online.sage_dashboard import export_dashboard
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"dashboard.html"
            export_dashboard({"title":"示例用户 <script> · __GRAPH__","nodes":[]},path)
            rendered=path.read_text()
            self.assertIn("<h1>示例用户 &lt;script&gt; · __GRAPH__</h1>",rendered)
            self.assertNotIn("陶小林",rendered)

    def test_prefix_cannot_be_reported_as_official_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            write_json(root/"protocol.json",{})
            write_json(root/"frozen.json",{"smoke":True})
            with patch("pmm_bench.pmm_benchmark.verify_run_code"), patch("pmm_bench.pmm_benchmark.require_llm"):
                with self.assertRaisesRegex(ValueError,"Official QA"):
                    evaluate(root,root,"fixture")

    def test_revision_keeps_original_manifest_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            write_json(root/"run_manifest.json",{"code_sha256":{}})
            write_json(root/"code_revision.json",{"original_manifest_sha256":file_hash(root/"run_manifest.json"),"code_sha256":{}})
            verify_run_code(root)
            write_json(root/"run_manifest.json",{"code_sha256":{},"changed":True})
            with self.assertRaisesRegex(ValueError,"original manifest"):
                verify_run_code(root)

    def test_dred_revision_requires_and_records_explicit_reason(self):
        self.check_resume_revision("dred")

    def test_sage_revision_requires_and_records_explicit_reason(self):
        self.check_resume_revision("sage")

    def check_resume_revision(self, stage):
        from pmm_bench.pmm_benchmark import main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root/"build"
            run.mkdir()
            write_json(run/"run_manifest.json", {"code_sha256":{}})
            original = (run/"run_manifest.json").read_bytes()
            args = ["pmm", "--package", tmp, "--output", tmp, "--stage", stage, "--record-code-revision"]
            with patch("sys.argv", args), patch("pmm_bench.pmm_benchmark."+stage) as fast:
                with self.assertRaisesRegex(ValueError, "revision-reason"):
                    main()
                fast.assert_not_called()
            self.assertFalse((run/"code_revision.json").exists())
            reason = "Performance-only resume; event algorithm and input unchanged"
            with patch("sys.argv", args+["--revision-reason", reason]), patch("pmm_bench.pmm_benchmark."+stage) as fast:
                main()
                fast.assert_called_once()
                self.assertEqual(fast.call_args.args[0], run)
            self.assertEqual((run/"run_manifest.json").read_bytes(), original)
            self.assertEqual(json.loads((run/"code_revision.json").read_text())["reason"], reason)
            verify_run_code(run)

    def test_raw_envelope_roundtrip_and_equal_timestamp_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = []
            for n in (11, 12, 13):
                raw = raw_record(content=f"observation {n}")
                records.append({"event_id":f"raw-line:{n}","user_id":"fixture","role":"user",
                    "content":json.dumps(raw),"timestamp":"2026-09-08T15:20:00.123+08:00",
                    "metadata":{"input_line":n,"source_lines":[n],"input_kind":"raw"}})
            source = root/"input.jsonl"
            source.write_text("\n".join(json.dumps(r) for r in records)+"\n")
            with EventMemory(root/"memory.sqlite") as m:
                report = import_jsonl(source,source_id="fixture",memory=m,input_format="pmm",user_id="fixture")
                observations = list(m.iter_observations())
                self.assertEqual([o.provenance.metadata["input_line"] for o in observations],[11,12,13])
                self.assertEqual([o.content for o in observations],[r["content"] for r in records])
            write_json(root/"import_report.json",report)
            self.assertEqual(verify_source(root)["status"],"passed")

    def test_context_respects_budget_and_counts_only_visible_sources(self):
        class Memory:
            def get_observation(self,oid):
                from datetime import datetime,timezone
                return SimpleNamespace(content="a"*1000 if oid=="large" else "测试内容",
                    start_time=datetime(2026,1,1,tzinfo=timezone.utc),
                    provenance=SimpleNamespace(metadata={"source_lines":[1 if oid=="large" else 2]}))
        retrieved={"events":[{"event_id":"E1","title":"事件","summary":"摘要",
                   "facts":[{"text":"事实","support":["large","small"]}]}],
                   "long_memories":[],"node_scores":{"E1":.8}}
        result=answer_context(retrieved,Memory(),char_limit=600)
        self.assertLessEqual(len(result["text"]),600)
        self.assertEqual(result["sent_raw_source_ids"],["raw-line:2"])
        self.assertEqual(len(result["omitted_raw_records"]),1)

    def test_judge_invalid_json_or_truncation_cannot_be_scored(self):
        response={"choices":[{"finish_reason":"stop","message":{"content":"not json"}}]}
        with self.assertRaises(ValueError):parse_judge(response)
        response["choices"][0]["message"]["content"]='{"label":"CORRECT","reason":"ok"}'
        self.assertEqual(parse_judge(response)["label"],"CORRECT")
        response["choices"][0]["finish_reason"]="length"
        with self.assertRaises(ValueError):parse_judge(response)

    def test_failed_questions_are_coverage_losses_not_zero_scores(self):
        report=summarize([{"status":"complete","category":"Single","score":1,"f1":.5},
                          {"status":"failed","category":"Single"}],{"Single":3})
        self.assertEqual(report["overall"]["accuracy"],1)
        self.assertEqual(report["overall"]["coverage"],1/3)
        self.assertEqual(report["overall"]["errors"],1)

    def test_abstention_counter_accepts_sentence_punctuation(self):
        records=[{"status":"complete","category":"Single","score":0,"f1":0,"prediction":s}
                 for s in ("证据不足，无法回答","证据不足，无法回答。\n","不能回答这个问题")]
        self.assertEqual(summarize(records,{"Single":3})["overall"]["abstentions"],2)


if __name__ == "__main__":unittest.main()
