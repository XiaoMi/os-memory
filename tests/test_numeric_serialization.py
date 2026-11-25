# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import unittest

from hypermem.schema.serialization import json_value, require_embedding, require_number
from hypermem.standardization import cosine


class NumericSerializationTests(unittest.TestCase):
    def test_json_numeric_fast_path_keeps_validation_and_copying(self):
        value = {'vector': [0., -0., 1, True, None, 'text'], 'nested': [{'x': 1.25}]}
        result = json_value(value)
        self.assertEqual(value, result)
        self.assertIsNot(value, result)
        self.assertIsNot(value['vector'], result['vector'])
        self.assertIsNot(value['nested'][0], result['nested'][0])
        for invalid in (float('nan'), float('inf'), -float('inf'), complex(1,2), (1,2), {1: 'bad'}):
            with self.subTest(value=repr(invalid)), self.assertRaises(ValueError):
                json_value(invalid)

    def test_nested_dataclass_datetime_and_primitive_subclasses(self):
        @dataclass
        class Payload:
            vector: list
            time: datetime
        value = Payload([.5, 1.], datetime(2026,5,1,tzinfo=timezone.utc))
        self.assertEqual(json_value(value), {'vector': [.5,1.], 'time': '2026-05-01T00:00:00+00:00'})
        class Float(float):
            pass
        class Integer(int):
            pass
        self.assertEqual(json_value([Float(.5), Integer(2)]), [.5, 2])
        require_embedding([Float(.5), Integer(2)])
        with self.assertRaises(ValueError):
            require_embedding([Float('nan')])

    def test_vector_checks_match_scalar_validator(self):
        for value in (0, 1., -1., True, False, None, '1', float('nan'), float('inf'), {}, []):
            try:
                require_number(value, 'embedding value')
            except ValueError as error:
                with self.subTest(value=repr(value)), self.assertRaisesRegex(ValueError, str(error)):
                    require_embedding([value])
            else:
                require_embedding([value])
        require_embedding(None)
        for value in ([], (1.,), 'vector', 1.):
            with self.assertRaises(ValueError):
                require_embedding(value)

    def test_cosine_remains_bit_exact_to_prior_formula(self):
        def prior(left, right):
            def unit(vector):
                for v in vector:
                    require_number(v, 'embedding value')
                length = math.sqrt(sum(float(v)**2 for v in vector))
                return [float(v)/length if length else 0. for v in vector]
            return sum(a*b for a,b in zip(unit(left), unit(right)))
        for left,right in (([0.]*1024,[1.]*1024), ([i/1024 for i in range(1024)], [(-1.)**i for i in range(1024)])):
            self.assertEqual(cosine(left,right), prior(left,right))
