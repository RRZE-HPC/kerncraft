#!/usr/bin/env python3
"""
High-level tests for the overall functionallity and things in kc.py
"""
import collections
import os
import unittest

from kerncraft import likwid_bench_auto as lba
from kerncraft import __version__ as kerncraft_version
from kerncraft.prefixedunit import PrefixedUnit


class TestLikwidBenchAuto(unittest.TestCase):
    @staticmethod
    def _find_file(name):
        testdir = os.path.dirname(__file__)
        name = os.path.join(testdir, 'test_files', name)
        assert os.path.exists(name)
        return name

    def test_get_machine_topology(self):
        # patch environment to include dummy likwid
        environ_orig = os.environ
        os.environ['PATH'] = self._find_file('dummy_likwid') + ':' + os.environ['PATH']

        self.maxDiff = None
        self.assertEqual(
            lba.get_machine_topology(cpuinfo_path=self._find_file('cpuinfo')),
            {'kerncraft version': kerncraft_version,
             'FLOPs per cycle': {'DP': {'ADD': 'INFORMATION_REQUIRED',
                                        'FMA': 'INFORMATION_REQUIRED',
                                        'MUL': 'INFORMATION_REQUIRED',
                                        'total': 'INFORMATION_REQUIRED'},
                                 'SP': {'ADD': 'INFORMATION_REQUIRED',
                                        'FMA': 'INFORMATION_REQUIRED',
                                        'MUL': 'INFORMATION_REQUIRED',
                                        'total': 'INFORMATION_REQUIRED'}},
             'NUMA domains per socket': 1,
             'cacheline size': 'INFORMATION_REQUIRED (in bytes, e.g. 64 B)',
             'clock': 'INFORMATION_REQUIRED (e.g., 2.7 GHz)',
             'compiler': collections.OrderedDict([
                 ('icc', 'INFORMATION_REQUIRED (e.g., -O3 -fno-alias -xAVX)'),
                 ('clang', 'INFORMATION_REQUIRED (e.g., -O3 -mavx, -D_POSIX_C_SOURCE=200112L'),
                 ('gcc', 'INFORMATION_REQUIRED (e.g., -O3 -march=ivybridge)')]),
             'cores per NUMA domain': 10,
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
                                   'cores per group': 1,
                                   'groups': 20,
                                   'level': 'L1',
                                   'performance counter metrics': {
                                       'accesses': 'INFORMATION_REQUIRED (e.g., '
                                                   'L1D_REPLACEMENT__PMC0)',
                                       'evicts': 'INFORMATION_REQUIRED (e.g., '
                                                 'L2_LINES_OUT_DIRTY_ALL__PMC2)',
                                       'misses': 'INFORMATION_REQUIRED (e.g., '
                                                 'L2_LINES_IN_ALL__PMC1)'},
                                   'size per group': PrefixedUnit(32.0, 'k', 'B'),
                                   'threads per group': 2},
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
                                   'cores per group': 1,
                                   'non-overlap upstream throughput': [
                                        'INFORMATION_REQUIRED (e.g. 24 B/cy)',
                                        'INFORMATION_REQUIRED (e.g. "half-duplex" or "full-duplex")'],
                                   'groups': 20,
                                   'level': 'L2',
                                   'performance counter metrics': {
                                       'accesses': 'INFORMATION_REQUIRED (e.g., '
                                                   'L1D_REPLACEMENT__PMC0)',
                                       'evicts': 'INFORMATION_REQUIRED (e.g., '
                                                 'L2_LINES_OUT_DIRTY_ALL__PMC2)',
                                       'misses': 'INFORMATION_REQUIRED (e.g., '
                                                 'L2_LINES_IN_ALL__PMC1)'},
                                   'size per group': PrefixedUnit(256.0, 'k', 'B'),
                                   'threads per group': 2},
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
                                   'cores per group': 10,
                                   'non-overlap upstream throughput': [
                                        'INFORMATION_REQUIRED (e.g. 24 B/cy)',
                                        'INFORMATION_REQUIRED (e.g. "half-duplex" or "full-duplex")'],
                                   'groups': 2,
                                   'level': 'L3',
                                   'performance counter metrics': {
                                       'accesses': 'INFORMATION_REQUIRED (e.g., '
                                                   'L1D_REPLACEMENT__PMC0)',
                                       'evicts': 'INFORMATION_REQUIRED (e.g., '
                                                 'L2_LINES_OUT_DIRTY_ALL__PMC2)',
                                       'misses': 'INFORMATION_REQUIRED (e.g., '
                                                 'L2_LINES_IN_ALL__PMC1)'},
                                   'size per group': PrefixedUnit(25.0, 'M', 'B'),
                                   'threads per group': 20},
                                  {'cores per group': 10,
                                   'non-overlap upstream throughput': [
                                        'full socket memory bandwidth',
                                        'INFORMATION_REQUIRED (e.g. "half-duplex" or "full-duplex")'],
                                   'level': 'MEM',
                                   'penalty cycles per read stream': 0,
                                   'size per group': None,
                                   'threads per group': 20}],
             'micro-architecture-modeler': 'INFORMATION_REQUIRED (options: OSACA, IACA)',
             'micro-architecture': 'INFORMATION_REQUIRED (options: NHM, WSM, SNB, IVB, HSW, BDW, SKL, SKX)',
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
