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
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

from hypermem.import_observations import import_jsonl
from hypermem.memory import EventMemory
from hypermem.source_verify import verify_source
from test_import_observations import raw_record


class SourceVerificationTests(unittest.TestCase):
    def make_run(self, root, limit=None):
        source = root / 'input.jsonl'
        source.write_text('\n'.join(json.dumps(raw_record(ts=1788852000123 + i)) for i in range(3)) + '\n')
        with EventMemory(root / 'memory.sqlite') as memory:
            report = import_jsonl(source, source_id='fixture', memory=memory, limit=limit)
        (root / 'import_report.json').write_text(json.dumps(report))
        return source

    def test_full_source_remains_full_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_run(root)
            result = verify_source(root)
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(result['fingerprint_scope'], 'full_file')
            self.assertEqual(result['checked_observations'], 3)

    def test_prefix_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_run(root, limit=2)
            self.assertEqual(verify_source(root)['status'], 'failed')
            result = verify_source(root, allow_prefix=True)
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(result['fingerprint_scope'], 'scanned_prefix')
            self.assertEqual(result['checked_observations'], 2)
            self.assertFalse(result['scan_complete'])

    def test_modified_source_prefix_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_run(root, limit=2)
            source.write_text(source.read_text().replace('Test App', 'Changed App'))
            self.assertEqual(verify_source(root, allow_prefix=True)['status'], 'failed')

    def test_window_preserves_original_line_ids_and_requires_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input.jsonl'
            base = 1788852000123
            source.write_text('\n'.join(json.dumps(raw_record(ts=base+i)) for i in range(4))+'\n')
            start = datetime.fromtimestamp((base+1)/1000,timezone.utc)
            end = datetime.fromtimestamp((base+3)/1000,timezone.utc)
            with EventMemory(root/'memory.sqlite') as memory:
                report = import_jsonl(source, source_id='fixture', memory=memory, start_time=start, end_time=end)
                self.assertEqual([o.provenance.metadata['line_number'] for o in memory.iter_observations()],[2,3])
            (root/'import_report.json').write_text(json.dumps(report))
            self.assertEqual(report['outside_time_window'],2)
            self.assertTrue(report['scan_complete'])
            self.assertEqual(verify_source(root)['status'],'failed')
            result = verify_source(root,allow_window=True)
            self.assertEqual(result['status'],'passed')
            self.assertEqual(result['fingerprint_scope'],'full_file')
            self.assertEqual(result['selection_scope'],'time_window')
            self.assertEqual(result['checked_observations'],2)


if __name__ == '__main__':
    unittest.main()
