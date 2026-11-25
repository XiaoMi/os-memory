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
import sys
import tempfile
import unittest
from unittest.mock import patch

# final now lives in the eval tree (pmm_bench); its STAGES import reaches the
# memory system's hypermem/ package at the repository root.
ROOT = Path(__file__).resolve().parents[1]
HYPERMEM_SOURCE = ROOT.parent
sys.path[:0] = [str(ROOT), str(HYPERMEM_SOURCE)]

from pmm_bench.final import FinalContractError, description, main, validate_final_run  # noqa: E402


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def complete_run(root: Path) -> Path:
    run = root / "build"
    run.mkdir(parents=True)
    write_json(run / "config.json", {})
    write_json(run / "sage_config.json", {"version": "hyperos-sage-v1"})
    write_json(run / "dred_audit.json", {"status": "passed"})
    write_json(run / "sage_report.json", {"schema": "hyperos-sage-tools-v1", "status": "complete",
                                          "structural_audit": {"status": "passed"}})
    write_json(run / "frozen.json", {"memory_sha256": "0"})
    return run


class FinalImplementationTests(unittest.TestCase):
    def test_description_names_dred_and_sage(self):
        d = description()
        self.assertEqual(d["method"], "DRED + SAGE")
        self.assertEqual(d["fast_path"]["name"], "DRED")
        self.assertEqual(d["slow_path"]["name"], "SAGE")
        self.assertEqual(d["stages"], ["prepare", "dred", "sage", "dashboard", "evaluate"])

    def test_complete_run_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(validate_final_run(complete_run(Path(tmp)))["status"], "passed")

    def test_skipped_failures_pass_and_are_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = complete_run(Path(tmp))
            write_json(run / "sage_report.json", {"schema": "hyperos-sage-tools-v1",
                                                  "status": "complete_with_skipped_failures",
                                                  "structural_audit": {"status": "passed"},
                                                  "skipped_failures": {"boundary": [{}], "sage": [{}, {}]}})
            self.assertEqual(validate_final_run(run)["skipped_failures"], {"boundary": 1, "sage": 2})

    def test_incomplete_sage_is_rejected(self):
        for change in ({"status": "needs_review"}, {"schema": "other"}, {"structural_audit": {"status": "failed"}}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                run = complete_run(Path(tmp))
                report = json.loads((run / "sage_report.json").read_text())
                write_json(run / "sage_report.json", {**report, **change})
                with self.assertRaises(FinalContractError):
                    validate_final_run(run)

    def test_unfrozen_or_failed_dred_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = complete_run(Path(tmp))
            (run / "frozen.json").unlink()
            with self.assertRaises(FinalContractError):
                validate_final_run(run)
        with tempfile.TemporaryDirectory() as tmp:
            run = complete_run(Path(tmp))
            write_json(run / "dred_audit.json", {"status": "failed"})
            with self.assertRaises(FinalContractError):
                validate_final_run(run)

    def test_run_delegates_and_records_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = complete_run(Path(tmp))
            with patch("pmm_bench.pmm_benchmark.main") as delegated:
                main(["run", "--package", tmp, "--output", tmp, "--user", "U01", "--stage", "dashboard"])
            delegated.assert_called_once()
            contract = json.loads((run / "final_run_contract.json").read_text())
            self.assertEqual(contract["method"], "DRED + SAGE")
            self.assertEqual(contract["entrypoint"], "pmm_bench.final")

    def test_evaluate_requires_a_complete_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = complete_run(Path(tmp))
            (run / "sage_report.json").unlink()
            with patch("pmm_bench.pmm_benchmark.main") as delegated, self.assertRaises(FinalContractError):
                main(["run", "--package", tmp, "--output", tmp, "--user", "U01", "--stage", "evaluate"])
            delegated.assert_not_called()


if __name__ == "__main__":
    unittest.main()
