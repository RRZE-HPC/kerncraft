"""
High-level tests for the overall functionallity and things in kc.py
"""
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

from kerncraft import likwid_bench_auto as lba
from kerncraft.prefixedunit import PrefixedUnit


class TestLikwidBenchAuto(unittest.TestCase):
    def _find_file(self, name):
        testdir = os.path.dirname(__file__)
        name = os.path.join(testdir, 'test_files', name)
        assert os.path.exists(name)
        return name

    def test_get_machine_topology(self):
        # patch environment to include dummy likwid
        environ_orig = os.environ
        os.environ['PATH'] = self._find_file('dummy_likwid')+':'+os.environ['PATH']

        self.maxDiff = None
        self.assertEqual(
            lba.get_machine_topology(cpuinfo_path=self._find_file('cpuinfo')),
            {'FLOPs per cycle': {'DP': {'ADD': 'INFORMATION_REQUIRED',
                                        'FMA': 'INFORMATION_REQUIRED',
                                        'MUL': 'INFORMATION_REQUIRED',
                                        'total': 'INFORMATION_REQUIRED'},
                                 'SP': {'ADD': 'INFORMATION_REQUIRED',
                                        'FMA': 'INFORMATION_REQUIRED',
                                        'MUL': 'INFORMATION_REQUIRED',
                                        'total': 'INFORMATION_REQUIRED'}},
             'NUMA domains per socket': 1.0,
             'cacheline size': 'INFORMATION_REQUIRED (in bytes, e.g. 64 B)',
             'clock': 'INFORMATION_REQUIRED (e.g., 2.7 GHz)',
             'compiler': {'clang': ['INFORMATION_REQUIRED (e.g., -O3 -mavx, -D_POSIX_C_SOURCE=200112L'],
                          'gcc': ['INFORMATION_REQUIRED (e.g., -O3 -march=ivybridge)'],
                          'icc': ['INFORMATION_REQUIRED (e.g., -O3 -fno-alias -xAVX)']},
             'cores per NUMA domain': 0.1,
             'cores per socket': 10,
             'memory hierarchy': [{'cache per group': {'cl_size': 'INFORMATION_REQUIRED '
                                                                  '(sets*ways*cl_size=32.00 kB)',
                                                       'load_from': 'L2',
                                                       'replacement_policy': 'INFORMATION_REQUIRED (options: '
                                                                             'LRU, FIFO, MRU, RR)',
                                                       'sets': 'INFORMATION_REQUIRED (sets*ways*cl_size=32.00 '
                                                               'kB)',
                                                       'store_to': 'L2',
                                                       'ways': 'INFORMATION_REQUIRED (sets*ways*cl_size=32.00 '
                                                               'kB)',
                                                       'write_allocate': 'INFORMATION_REQUIRED (True/False)',
                                                       'write_back': 'INFORMATION_REQUIRED (True/False)'},
                                   'cores per group': 1.0,
                                   'cycles per cacheline transfer': 'INFORMATION_REQUIRED',
                                   'groups': 20,
                                   'level': 'L1',
                                   'performance counter metrics': {'accesses': 'INFORMATION_REQUIRED (e.g., '
                                                                               'L1D_REPLACEMENT__PMC0)',
                                                                   'evicts': 'INFORMATION_REQUIRED (e.g., '
                                                                             'L2_LINES_OUT_DIRTY_ALL__PMC2)',
                                                                   'misses': 'INFORMATION_REQUIRED (e.g., '
                                                                             'L2_LINES_IN_ALL__PMC1)'},
                                   'size per group': PrefixedUnit(32.0, 'k', 'B'),
                                   'threads per group': 2.0},
                                  {'cache per group': {'cl_size': 'INFORMATION_REQUIRED '
                                                                  '(sets*ways*cl_size=256.00 kB)',
                                                       'load_from': 'L3',
                                                       'replacement_policy': 'INFORMATION_REQUIRED (options: '
                                                                             'LRU, FIFO, MRU, RR)',
                                                       'sets': 'INFORMATION_REQUIRED (sets*ways*cl_size=256.00 '
                                                               'kB)',
                                                       'store_to': 'L3',
                                                       'ways': 'INFORMATION_REQUIRED (sets*ways*cl_size=256.00 '
                                                               'kB)',
                                                       'write_allocate': 'INFORMATION_REQUIRED (True/False)',
                                                       'write_back': 'INFORMATION_REQUIRED (True/False)'},
                                   'cores per group': 1.0,
                                   'cycles per cacheline transfer': 'INFORMATION_REQUIRED',
                                   'groups': 20,
                                   'level': 'L2',
                                   'performance counter metrics': {'accesses': 'INFORMATION_REQUIRED (e.g., '
                                                                               'L1D_REPLACEMENT__PMC0)',
                                                                   'evicts': 'INFORMATION_REQUIRED (e.g., '
                                                                             'L2_LINES_OUT_DIRTY_ALL__PMC2)',
                                                                   'misses': 'INFORMATION_REQUIRED (e.g., '
                                                                             'L2_LINES_IN_ALL__PMC1)'},
                                   'size per group': PrefixedUnit(256.0, 'k', 'B'),
                                   'threads per group': 2.0},
                                  {'cache per group': {'cl_size': 'INFORMATION_REQUIRED '
                                                                  '(sets*ways*cl_size=25.00 MB)',
                                                       'replacement_policy': 'INFORMATION_REQUIRED (options: '
                                                                             'LRU, FIFO, MRU, RR)',
                                                       'sets': 'INFORMATION_REQUIRED (sets*ways*cl_size=25.00 '
                                                               'MB)',
                                                       'ways': 'INFORMATION_REQUIRED (sets*ways*cl_size=25.00 '
                                                               'MB)',
                                                       'write_allocate': 'INFORMATION_REQUIRED (True/False)',
                                                       'write_back': 'INFORMATION_REQUIRED (True/False)'},
                                   'cores per group': 10.0,
                                   'cycles per cacheline transfer': 'INFORMATION_REQUIRED',
                                   'groups': 2,
                                   'level': 'L3',
                                   'performance counter metrics': {'accesses': 'INFORMATION_REQUIRED (e.g., '
                                                                               'L1D_REPLACEMENT__PMC0)',
                                                                   'evicts': 'INFORMATION_REQUIRED (e.g., '
                                                                             'L2_LINES_OUT_DIRTY_ALL__PMC2)',
                                                                   'misses': 'INFORMATION_REQUIRED (e.g., '
                                                                             'L2_LINES_IN_ALL__PMC1)'},
                                   'size per group': PrefixedUnit(25.0, 'M', 'B'),
                                   'threads per group': 20.0},
                                  {'cores per group': 10,
                                   'cycles per cacheline transfer': None,
                                   'level': 'MEM',
                                   'penalty cycles per read stream': 0,
                                   'size per group': None,
                                   'threads per group': 20}],
             'micro-architecture': 'INFORMATION_REQUIRED (options: NHM, WSM, SNB, IVB, HSW)',
             'model name': 'Intel(R) Xeon(R) CPU E5-2660 v2 @ 2.20GHz',
             'model type': 'Intel Xeon IvyBridge EN/EP/EX processor',
             'non-overlapping model': {'performance counter metric': 'INFORAMTION_REQUIRED '
                                                                     'Example:max(UOPS_DISPATCHED_PORT_PORT_0__PMC2, '
                                                                     'UOPS_DISPATCHED_PORT_PORT_1__PMC3,    '
                                                                     'UOPS_DISPATCHED_PORT_PORT_4__PMC0, '
                                                                     'UOPS_DISPATCHED_PORT_PORT_5__PMC1)',
                                       'ports': 'INFORAMTION_REQUIRED (list of ports as they appear in IACA, '
                                                'e.g.), ["0", "0DV", "1", "2", "2D", "3", "3D", "4", "5", "6", '
                                                '"7"])'},
             'overlapping model': {'performance counter metric': 'INFORAMTION_REQUIRED '
                                                                 'Example:max(UOPS_DISPATCHED_PORT_PORT_0__PMC2, '
                                                                 'UOPS_DISPATCHED_PORT_PORT_1__PMC3,    '
                                                                 'UOPS_DISPATCHED_PORT_PORT_4__PMC0, '
                                                                 'UOPS_DISPATCHED_PORT_PORT_5__PMC1)',
                                   'ports': 'INFORAMTION_REQUIRED (list of ports as they appear in IACA, '
                                            'e.g.), ["0", "0DV", "1", "2", "2D", "3", "3D", "4", "5", "6", '
                                            '"7"])'},
             'sockets': 2,
             'threads per core': 2})


        # restore enviornment
        os.environ = environ_orig


if __name__ == '__main__':
    unittest.main()
