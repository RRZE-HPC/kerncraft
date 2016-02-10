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
from itertools import chain

import six
import sympy
from ruamel import yaml

sys.path.insert(0, '..')
from kerncraft.kernel import Kernel, KernelCode, KernelDescription


class TestKernel(unittest.TestCase):
    def setUp(self):
        self.twod_code = open(self._find_file('2d-5pt.c')).read()
        self.threed_code = open(self._find_file('3d-7pt.c')).read()
        self.twod_description = yaml.load(open(self._find_file('2d-5pt.yml')).read())
       
    def _find_file(self, name):
        testdir = os.path.dirname(__file__)
        name = os.path.join(testdir, 'test_files', name)
        assert os.path.exists(name)
        return name
        
    def test_array_sizes_2d(self):
        k = KernelCode(self.twod_code)
        k.set_constant('N', 10)
        k.set_constant('M', 20)
        sizes = k.array_sizes(in_bytes=True, subs_consts=True)
        # 8 byte per double
        checked_sizes = {'a': 20*10*8, 'b': 20*10*8}
        self.assertEqual(sizes, checked_sizes)
    
    def test_array_sizes_3d(self):
        k = KernelCode(self.threed_code)
        k.set_constant('N', 10)
        k.set_constant('M', 20)
        sizes = k.array_sizes(in_bytes=True, subs_consts=True)
        # 8 byte per double
        checked_sizes = {'a': 20*10*10*8, 'b': 20*10*10*8}
        self.assertEqual(sizes, checked_sizes)
    
    def test_global_offsets_2d(self):
        k = KernelCode(self.twod_code)
        k.set_constant('N', 10)
        k.set_constant('M', 20)
        sizes = k.array_sizes(in_bytes=True, subs_consts=True)
        offsets = k.compile_global_offsets(iteration=0, spacing=0)
        read_offsets, write_offsets = list(offsets)[0]
        # read access to a[j][i-1], a[j][i+1], a[j-1][i], a[j+1][i]
        six.assertCountEqual(
            self,
            [(1*10+0)*8, (1*10+2)*8, (0*10+1)*8, (2*10+1)*8],
            read_offsets)
        # write access to b[i][j]
        six.assertCountEqual(
            self,
            [sizes['a']+(1*10+1)*8],
            write_offsets)
        
    def test_global_offsets_3d(self):
        k = KernelCode(self.threed_code)
        k.set_constant('N', 10)
        k.set_constant('M', 20)
        sizes = k.array_sizes(in_bytes=True, subs_consts=True)
        offsets = k.compile_global_offsets(iteration=0, spacing=0)
        read_offsets, write_offsets = list(offsets)[0]
        # read access to a[k][j][i], a[k][j][i-1], a[k][j][i+1], a[k][j-1][i],
        #                a[k][j+1][i], a[k+1][j][i], a[k-1][j][i]
        six.assertCountEqual(self,
                             [(1*10*10+1*10+1)*8, (1*10*10+1*10+0)*8, (1*10*10+1*10+2)*8,
                              (1*10*10+0*10+1)*8, (1*10*10+2*10+1)*8, (2*10*10+1*10+1)*8,
                              (0*10*10+1*10+1)*8],
                             read_offsets)
        # write access to b[i][j]
        six.assertCountEqual(self, [sizes['a']+(1*10*10+1*10+1)*8], write_offsets)

    def test_from_description(self):
        k_descr = KernelDescription(self.twod_description)
        k_code = KernelCode(self.twod_code)
        
        self.assertEqual(k_descr._flops, k_code._flops)
        self.assertEqual(k_descr._sources, k_code._sources)
        self.assertEqual(k_descr._destinations, k_code._destinations)
        self.assertEqual(k_descr.datatype, k_code.datatype)
        self.assertEqual(k_descr.variables, k_code.variables)
        self.assertEqual(k_descr._loop_stack, k_code._loop_stack)

if __name__ == '__main__':
    #unittest.main()
    suite = unittest.TestLoader().loadTestsFromTestCase(TestKernel)
    unittest.TextTestRunner(verbosity=2).run(suite)