'''
High-level tests for the overall functionallity and things in kc.py
'''
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import sys
import os
import unittest
import tempfile
import shutil
import pickle
from pprint import pprint
from io import StringIO
from distutils.spawn import find_executable
import platform

import six
import sympy

sys.path.insert(0, '..')
from kerncraft import kerncraft as kc
from kerncraft.prefixedunit import PrefixedUnit


class TestLayerCondition(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.temp_dir)

    def _find_file(self, name):
        testdir = os.path.dirname(__file__)
        name = os.path.join(testdir, 'test_files', name)
        assert os.path.exists(name)
        return name

    def test_kernel_requirements(self):
        output_stream = StringIO()
        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('phinally_gcc.yaml'),
                                  '-p', 'LC',
                                  self._find_file('2d-5pt-unrolled.c')])
        kc.check_arguments(args, parser)
        with self.assertRaises(ValueError) as cm:
            kc.run(parser, args, output_file=output_stream)


if __name__ == '__main__':
    unittest.main()