#!/usr/bin/env python3
"""
Tests for validity of example files.
"""
import os
from glob import glob
import unittest
from unittest import mock
from collections import OrderedDict

from kerncraft import machinemodel
from kerncraft.prefixedunit import PrefixedUnit
from kerncraft import __version__ as kerncraft_version



class TestMachineModel(unittest.TestCase):
    @staticmethod
    def _find_file(name):
        testdir = os.path.dirname(__file__)
        name = os.path.join(testdir, 'test_files', name)
        assert os.path.exists(name)
        return name

    def setUp(self):
        self.machine = machinemodel.MachineModel(self._find_file('SandyBridgeEP_E5-2680.yml'))

    def test_types(self):
        self.assertEqual(type(self.machine['clock']), PrefixedUnit)
        self.assertEqual(self.machine['clock'].base_value(), 2700000000.0)
        self.assertEqual(self.machine['clock'].unit, "Hz")

    @mock.patch('kerncraft.machinemodel.get_cpu_frequency', new=lambda: 2.2e9)
    def test_machine_model_update(self):
        # patch environment to include dummy likwid
        environ_orig = os.environ
        os.environ['PATH'] = self._find_file('dummy_likwid') + ':' + os.environ['PATH']

        m = machinemodel.MachineModel()
        m.update(readouts=True, memory_hierarchy=True, benchmarks=False, overwrite=True,
                 cpuinfo_path=self._find_file('cpuinfo'))

        self.maxDiff = None

        correct = {'kerncraft version': kerncraft_version,
                   'FLOPs per cycle': {'DP': {'ADD': 'INFORMATION_REQUIRED',
                                              'FMA': 'INFORMATION_REQUIRED',
                                              'MUL': 'INFORMATION_REQUIRED',
                                              'total': 'INFORMATION_REQUIRED'},
                                       'SP': {'ADD': 'INFORMATION_REQUIRED',
                                              'FMA': 'INFORMATION_REQUIRED',
                                              'MUL': 'INFORMATION_REQUIRED',
                                              'total': 'INFORMATION_REQUIRED'}},
                   'NUMA domains per socket': 1,
                   'benchmarks': 'INFORMATION_REQUIRED',
                   'cacheline size': 'INFORMATION_REQUIRED (in bytes, e.g. 64 B)',
                   'clock': PrefixedUnit(2200000000.0, '', 'Hz'),
                   'compiler': OrderedDict(
                       [('icc', 'INFORMATION_REQUIRED (e.g., -O3 -fno-alias -xAVX)'),
                        ('clang',
                         'INFORMATION_REQUIRED (e.g., -O3 -mavx, -D_POSIX_C_SOURCE=200112L, check `gcc -march=native -Q --help=target | '
                         'grep -- "-march="`)'),
                        ('gcc',
                         'INFORMATION_REQUIRED (e.g., -O3 -march=ivybridge, check `gcc -march=native -Q --help=target | grep -- '
                         '"-march="`)')]),
                   'cores per NUMA domain': 10,
                   'cores per socket': 10,
                   'in-core model': OrderedDict([('IACA',
                                                  'INFORMATION_REQUIRED (e.g., NHM, WSM, SNB, IVB, HSW, BDW, SKL, SKX)'),
                                                 ('OSACA',
                                                  'INFORMATION_REQUIRED (e.g., NHM, WSM, SNB, IVB, HSW, BDW, SKL, SKX)'),
                                                 ('LLVM-MCA',
                                                  'INFORMATION_REQUIRED (e.g., -mcpu=skylake-avx512)')]),
                   'memory hierarchy': [OrderedDict([('level', 'L1'),
                                                     ('performance counter metrics',
                                                      {
                                                          'accesses': 'INFORMATION_REQUIRED (e.g., L1D_REPLACEMENT__PMC0)',
                                                          'evicts': 'INFORMATION_REQUIRED (e.g., L2_LINES_OUT_DIRTY_ALL__PMC2)',
                                                          'misses': 'INFORMATION_REQUIRED (e.g., L2_LINES_IN_ALL__PMC1)'}),
                                                     ('cache per group',
                                                      OrderedDict([('sets',
                                                                    'INFORMATION_REQUIRED (sets*ways*cl_size=32.00 kB)'),
                                                                   ('ways',
                                                                    'INFORMATION_REQUIRED (sets*ways*cl_size=32.00 kB)'),
                                                                   ('cl_size',
                                                                    'INFORMATION_REQUIRED (sets*ways*cl_size=32.00 kB)'),
                                                                   ('replacement_policy',
                                                                    'INFORMATION_REQUIRED (options: LRU, FIFO, MRU, RR)'),
                                                                   ('write_allocate',
                                                                    'INFORMATION_REQUIRED (True/False)'),
                                                                   ('write_back',
                                                                    'INFORMATION_REQUIRED (True/False)'),
                                                                   ('load_from', 'L2'),
                                                                   ('store_to', 'L2')])),
                                                     ('size per group',
                                                      PrefixedUnit(32.0, 'k', 'B')),
                                                     ('groups', 20),
                                                     ('cores per group', 1),
                                                     ('threads per group', 2)]),
                                        OrderedDict([('level', 'L2'),
                                                     ('upstream throughput',
                                                      ['INFORMATION_REQUIRED (e.g. 24 B/cy)',
                                                       'INFORMATION_REQUIRED (e.g. "half-duplex" or "full-duplex")']),
                                                     ('performance counter metrics',
                                                      {
                                                          'accesses': 'INFORMATION_REQUIRED (e.g., L1D_REPLACEMENT__PMC0)',
                                                          'evicts': 'INFORMATION_REQUIRED (e.g., L2_LINES_OUT_DIRTY_ALL__PMC2)',
                                                          'misses': 'INFORMATION_REQUIRED (e.g., L2_LINES_IN_ALL__PMC1)'}),
                                                     ('cache per group',
                                                      OrderedDict([('sets',
                                                                    'INFORMATION_REQUIRED (sets*ways*cl_size=256.00 kB)'),
                                                                   ('ways',
                                                                    'INFORMATION_REQUIRED (sets*ways*cl_size=256.00 kB)'),
                                                                   ('cl_size',
                                                                    'INFORMATION_REQUIRED (sets*ways*cl_size=256.00 kB)'),
                                                                   ('replacement_policy',
                                                                    'INFORMATION_REQUIRED (options: LRU, FIFO, MRU, RR)'),
                                                                   ('write_allocate',
                                                                    'INFORMATION_REQUIRED (True/False)'),
                                                                   ('write_back',
                                                                    'INFORMATION_REQUIRED (True/False)'),
                                                                   ('load_from', 'L3'),
                                                                   ('store_to', 'L3')])),
                                                     ('size per group',
                                                      PrefixedUnit(256.0, 'k', 'B')),
                                                     ('groups', 20),
                                                     ('cores per group', 1),
                                                     ('threads per group', 2)]),
                                        OrderedDict([('level', 'L3'),
                                                     ('upstream throughput',
                                                      ['INFORMATION_REQUIRED (e.g. 24 B/cy)',
                                                       'INFORMATION_REQUIRED (e.g. "half-duplex" or "full-duplex")']),
                                                     ('performance counter metrics',
                                                      {
                                                          'accesses': 'INFORMATION_REQUIRED (e.g., L1D_REPLACEMENT__PMC0)',
                                                          'evicts': 'INFORMATION_REQUIRED (e.g., L2_LINES_OUT_DIRTY_ALL__PMC2)',
                                                          'misses': 'INFORMATION_REQUIRED (e.g., L2_LINES_IN_ALL__PMC1)'}),
                                                     ('cache per group',
                                                      OrderedDict([('sets',
                                                                    'INFORMATION_REQUIRED (sets*ways*cl_size=25.00 MB)'),
                                                                   ('ways',
                                                                    'INFORMATION_REQUIRED (sets*ways*cl_size=25.00 MB)'),
                                                                   ('cl_size',
                                                                    'INFORMATION_REQUIRED (sets*ways*cl_size=25.00 MB)'),
                                                                   ('replacement_policy',
                                                                    'INFORMATION_REQUIRED (options: LRU, FIFO, MRU, RR)'),
                                                                   ('write_allocate',
                                                                    'INFORMATION_REQUIRED (True/False)'),
                                                                   ('write_back',
                                                                    'INFORMATION_REQUIRED (True/False)')])),
                                                     ('size per group',
                                                      PrefixedUnit(25.0, 'M', 'B')),
                                                     ('groups', 2),
                                                     ('cores per group', 10),
                                                     ('threads per group', 20)]),
                                        OrderedDict([('level', 'MEM'),
                                                     ('cores per group', 10),
                                                     ('threads per group', 20),
                                                     ('upstream throughput',
                                                      ['full socket memory bandwidth',
                                                       'INFORMATION_REQUIRED (e.g. "half-duplex" or "full-duplex")']),
                                                     ('penalty cycles per cacheline load', 0),
                                                     ('penalty cycles per cacheline store', 0),
                                                     ('size per group', None)])],
                   'model name': 'Intel(R) Xeon(R) CPU E5-2660 v2 @ 2.20GHz',
                   'model type': 'Intel Xeon IvyBridge EN/EP/EX processor',
                   'non-overlapping model': {
                       'performance counter metric': 'INFORMATION_REQUIRED Example:max(UOPS_DISPATCHED_PORT_PORT_0__PMC2, '
                                                     'UOPS_DISPATCHED_PORT_PORT_1__PMC3,    UOPS_DISPATCHED_PORT_PORT_4__PMC0, '
                                                     'UOPS_DISPATCHED_PORT_PORT_5__PMC1)',
                       'ports': 'INFORMATION_REQUIRED (list of ports as they appear in IACA, e.g.,, ["0", "0DV", "1", "2", "2D", "3", '
                                '"3D", "4", "5", "6", "7"])'},
                   'overlapping model': {
                       'performance counter metric': 'INFORMATION_REQUIRED Example:max(UOPS_DISPATCHED_PORT_PORT_0__PMC2, '
                                                     'UOPS_DISPATCHED_PORT_PORT_1__PMC3,    UOPS_DISPATCHED_PORT_PORT_4__PMC0, '
                                                     'UOPS_DISPATCHED_PORT_PORT_5__PMC1)',
                       'ports': 'INFORMATION_REQUIRED (list of ports as they appear in IACA, e.g.,, ["0", "0DV", "1", "2", "2D", "3", "3D", '
                                '"4", "5", "6", "7"])'},
                   'sockets': 2,
                   'threads per core': 2}

        for k in correct:
            self.assertEqual(m[k], correct[k])

        # restore enviornment
        os.environ = environ_orig
