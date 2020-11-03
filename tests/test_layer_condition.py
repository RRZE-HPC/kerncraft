#!/usr/bin/env python3
"""
High-level tests for the overall functionallity and things in kc.py
"""
import os
import unittest
import tempfile
import shutil
import pickle
from io import StringIO

import sympy
from sympy import oo

from kerncraft import kerncraft as kc
from kerncraft.prefixedunit import PrefixedUnit


def recursive_dict_get(d, key_path):
    """Return element at key_path (tuple) in recursive dictionary d."""
    if len(key_path) > 1:
        return recursive_dict_get(d[key_path[0]], key_path[1:])
    else:
        return d[key_path[0]]


class TestLayerCondition(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.maxDiff = None

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.temp_dir)

    @staticmethod
    def _find_file(name):
        test_dir = os.path.dirname(__file__)
        name = os.path.join(test_dir, 'test_files', name)
        assert os.path.exists(name)
        return name

    def test_kernel_requirements(self):
        output_stream = StringIO()
        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('SandyBridgeEP_E5-2680.yml'),
                                  '-p', 'LC',
                                  self._find_file('2d-5pt-unrolled.c')])
        kc.check_arguments(args, parser)
        with self.assertRaises(ValueError) as cm:
            kc.run(parser, args, output_file=output_stream)

    def test_3d_7pt(self):
        store_file = os.path.join(self.temp_dir, 'test_3d7pt_LC.pickle')
        output_stream = StringIO()
        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('SandyBridgeEP_E5-2680.yml'),
                                  '-p', 'LC',
                                  self._find_file('3d-7pt.c'),
                                  '-D', '.', '0',
                                  '-vvv',
                                  '--store', store_file])
        kc.check_arguments(args, parser)
        kc.run(parser, args, output_file=output_stream)
        with open(store_file, 'rb') as f:
            results = pickle.load(f)
        result = next(iter(results.values()))
        N, M, i, j, k = sympy.var('N, M, i, j, k')
        result_expected = {'accesses':
                     {'a': [(k - 1, j, i),
                            (k, j - 1, i),
                            (k, j, i - 1),
                            (k, j, i),
                            (k, j, i + 1),
                            (k, j + 1, i),
                            (k + 1, j, i)],
                      'b': [(k, j, i)],
                      's': []},
         'cache': [[{'condition': M*N**2 < 2048,
                     'evicts': 0,
                     'hits': 8,
                     'misses': 0,
                     'tail': oo},
                    {'condition': 2*N**2 - N <= 2048,
                     'evicts': 1,
                     'hits': 6,
                     'misses': 2,
                     'tail': 8*N**2 - 8*N},
                    {'condition': 3*N <= 2050,
                     'evicts': 1,
                     'hits': 4,
                     'misses': 4,
                     'tail': 8 * N - 8},
                    {'condition': True,
                     'evicts': 1,
                     'hits': 2,
                     'misses': 6,
                     'tail': 8}],
                   [{'condition': M*N**2 < 16384,
                     'evicts': 0,
                     'hits': 8,
                     'misses': 0,
                     'tail': oo},
                    {'condition': 2*N**2 - N <= 16384,
                     'evicts': 1,
                     'hits': 6,
                     'misses': 2,
                     'tail': 8 * N ** 2 - 8 * N},
                    {'condition': N <= 5462,
                     'evicts': 1,
                     'hits': 4,
                     'misses': 4,
                     'tail': 8 * N - 8},
                    {'condition': True,
                     'evicts': 1,
                     'hits': 2,
                     'misses': 6,
                     'tail': 8}],
                   [{'condition': M*N**2 < 1310720,
                     'evicts': 0,
                     'hits': 8,
                     'misses': 0,
                     'tail': oo},
                    {'condition': 2*N**2 - N <= 1310720,
                     'evicts': 1,
                     'hits': 6,
                     'misses': 2,
                     'tail': 8 * N ** 2 - 8 * N},
                    {'condition': 3*N <= 1310722,
                     'evicts': 1,
                     'hits': 4,
                     'misses': 4,
                     'tail': 8 * N - 8},
                    {'condition': True,
                     'evicts': 1,
                     'hits': 2,
                     'misses': 6,
                     'tail': 8}]],
         'destinations': {('b', (k, j, i))},
         'distances': [oo, oo, N * (N - 1), N * (N - 1), N - 1, N - 1, 1, 1],
         'distances_bytes': [oo, oo, 8 * N * (N - 1), 8 * N * (N - 1), 8 * N - 8, 8 * N - 8, 8, 8]}
        # Iterate over expected results and validate with generated results
        stack = [((k,), v) for k, v in result_expected.items()]
        while stack:
            key_path, value = stack.pop()
            if isinstance(value, dict):
                stack.extend([(key_path + (k,), v) for k, v in value.items()])
            else:
                self.assertEqual(value, recursive_dict_get(result, key_path),
                                 msg="at key_path={}".format(key_path))

    def test_constantdim(self):
        store_file = os.path.join(self.temp_dir, 'test_constantdim_LC.pickle')
        output_stream = StringIO()
        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('SandyBridgeEP_E5-2680.yml'),
                                  '-p', 'LC',
                                  self._find_file('constantdim.c'),
                                  '-D', 'N', '1224',
                                  '-D', 'M', '1224',
                                  '-vvv',
                                  '--store', store_file])
        kc.check_arguments(args, parser)
        kc.run(parser, args, output_file=output_stream)
        with open(store_file, 'rb') as f:
            results = pickle.load(f)
        result = next(iter(results.values()))

        N, M, j, i = sympy.var('N'), sympy.var('M'), sympy.var('j'), sympy.var('i')
        result_expected = \
            {'accesses': {'W': [(j, i), (1, j, i)],
                          'a': [(j - 1, i), (j, i - 1), (j, i), (j, i + 1), (j + 1, i)],
                          'b': [(j, i)]},
             'cache': [[{'condition': M*N < 1024,
                         'evicts': 0,
                         'hits': 8,
                         'misses': 0,
                         'tail': oo},
                        {'condition': 2*M*N + N <= 2048,
                         'evicts': 1,
                         'hits': 5,
                         'misses': 3,
                         'tail': 8 * M * N},
                        {'condition': 3*N <= 2050,
                         'evicts': 1,
                         'hits': 4,
                         'misses': 4,
                         'tail': 8*N - 8},
                        {'condition': True,
                         'evicts': 1,
                         'hits': 2,
                         'misses': 6,
                         'tail': 8}],
                       [{'condition': M*N < 8192,
                         'evicts': 0,
                         'hits': 8,
                         'misses': 0,
                         'tail': oo},
                        {'condition': 2*M*N + N <= 16384,
                         'evicts': 1,
                         'hits': 5,
                         'misses': 3,
                         'tail': 8 * M * N},
                        {'condition':  N <= 5462,
                         'evicts': 1,
                         'hits': 4,
                         'misses': 4,
                         'tail': 8 * N - 8},
                        {'condition': True,
                         'evicts': 1,
                         'hits': 2,
                         'misses': 6,
                         'tail': 8}],
                       [{'condition': M*N < 655360,
                         'evicts': 0,
                         'hits': 8,
                         'misses': 0,
                         'tail': oo},
                        {'condition': 2*M*N + N <= 1310720,
                         'evicts': 1,
                         'hits': 5,
                         'misses': 3,
                         'tail': 8 * M * N},
                        {'condition': 3*N <= 1310722,
                         'evicts': 1,
                         'hits': 4,
                         'misses': 4,
                         'tail': 8 * N - 8},
                        {'condition': True,
                         'evicts': 1,
                         'hits': 2,
                         'misses': 6,
                         'tail': 8}]],
             'destinations': {('b', (j, i))},
             'distances': [oo, oo, oo, M * N, N - 1, N - 1, 1, 1],
             'distances_bytes': [oo, oo, oo, 8 * M * N, 8 * N - 8, 8 * N - 8, 8, 8]}
        # Iterate over expected results and validate with generated results
        stack = [((k,), v) for k, v in result_expected.items()]
        while stack:
            key_path, value = stack.pop()
            if isinstance(value, dict):
                stack.extend([(key_path + (k,), v) for k, v in value.items()])
            if isinstance(value, list):
                stack.extend([(key_path + (i,), v) for i, v in enumerate(value)])
            else:
                self.assertEqual(value, recursive_dict_get(result, key_path),
                                 msg="at key_path={}".format(key_path))


if __name__ == '__main__':
    unittest.main(buffer=True)
