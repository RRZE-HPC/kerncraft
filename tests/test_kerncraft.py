#!/usr/bin/env python3
"""
High-level tests for the overall functionallity and things in kc.py
"""
import os
import unittest
import tempfile
import shutil
import pickle
from distutils.spawn import find_executable
import platform
import warnings
import sys

import sympy

from kerncraft import kerncraft as kc
from kerncraft.prefixedunit import PrefixedUnit


def assertRelativlyEqual(actual, desired, rel_diff=0.0):
    """
    Test for relative difference between actual and desired

    passes if abs(actual-desired)/abs(desired) % 1.0 < rel_diff
    """
    if actual == desired:
        # Catching NaN, inf and 0
        return
    if not abs(actual - desired) / abs(desired) % 1.0 <= rel_diff:
        raise AssertionError("relative difference was not met with {}. Expected {!r} with rel. "
                             "difference of {!r}.".format(actual, desired, rel_diff))


def ignore_warnings(test_func):
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_func(self, *args, **kwargs)
    return do_test

class TestKerncraft(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.temp_dir)

    @staticmethod
    def _find_file(name):
        testdir = os.path.dirname(__file__)
        name = os.path.join(testdir, 'test_files', name)
        assert os.path.exists(name)
        return name

    def test_2d5pt_ECMData_SIM(self):
        store_file = os.path.join(self.temp_dir, 'test_2d5pt_ECMData.pickle')

        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('SandyBridgeEP_E5-2680.yml'),
                                  '-p', 'ECMData',
                                  self._find_file('2d-5pt.c'),
                                  '-D', 'N', '10000-100000:2log10',
                                  '-D', 'N', '1000',
                                  '-D', 'M', '50',
                                  '-vvv',
                                  '--store', store_file])
        kc.check_arguments(args, parser)
        kc.run(parser, args, output_file=sys.stdout)

        with open(store_file, 'rb') as f:
            results = pickle.load(f)

        # Check for correct variations of constants
        self.assertEqual(len(results), 3)

        # Check if results contains correct kernel and variations of constants
        for key in results:
            d = dict(key)
            self.assertEqual(d['code_file'].split('/')[-1], '2d-5pt.c')
            self.assertEqual(d['machine'].split('/')[-1], 'SandyBridgeEP_E5-2680.yml')
            self.assertEqual(d['pmodel'], 'ECMData')
        self.assertTrue(any([('define', (('M', 50), ('N', 1000))) in k for k in results]))
        self.assertTrue(any([('define', (('M', 50), ('N', 10000))) in k for k in results]))
        self.assertTrue(any([('define', (('M', 50), ('N', 100000))) in k for k in results]))

        # Output of first result:
        key = [k for k in results if ('define', (('M', 50), ('N', 1000))) in k][0]
        result = results[key]

        # 2 arrays * 1000*50 doubles/array * 8 Bytes/double = 781kB
        # -> fully cached in L3
        assertRelativlyEqual(result['L2'], 6, 0.05)
        assertRelativlyEqual(result['L3'], 6, 0.05)
        self.assertAlmostEqual(result['MEM'], 0.0, places=3)

    def test_2d5pt_ECMData_LC(self):
        store_file = os.path.join(self.temp_dir, 'test_2d5pt_ECMData.pickle')

        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('SandyBridgeEP_E5-2680.yml'),
                                  '-p', 'ECMData',
                                  self._find_file('2d-5pt.c'),
                                  '-D', 'N', '10000-100000:2log10',
                                  '-D', 'N', '1000',
                                  '-D', 'M', '50',
                                  '-vvv',
                                  '--cache-predictor=LC',
                                  '--store', store_file])
        kc.check_arguments(args, parser)
        kc.run(parser, args, output_file=sys.stdout)

        with open(store_file, 'rb') as f:
            results = pickle.load(f)

        # Check for correct variations of constants
        self.assertEqual(len(results), 3)

        # Check if results contains correct kernel and some other infoormation
        key = [k for k in results if ('define', (('M', 50), ('N', 1000))) in k][0]
        key_dict = dict(key)
        self.assertEqual(key_dict['code_file'].split('/')[-1], '2d-5pt.c')
        self.assertEqual(key_dict['machine'].split('/')[-1], 'SandyBridgeEP_E5-2680.yml')
        self.assertEqual(key_dict['asm_block'], 'auto')
        self.assertEqual(key_dict['pmodel'], 'ECMData')

        # Output of first result:
        result = results[key]

        # 2 arrays * 1000*50 doubles/array * 8 Bytes/double = 781kB
        # -> fully cached in L3
        assertRelativlyEqual(result['L2'], 6, 0.05)
        assertRelativlyEqual(result['L3'], 6, 0.05)
        self.assertAlmostEqual(result['MEM'], 0.0, places=2)

    def test_2d5pt_Roofline(self):
        store_file = os.path.join(self.temp_dir, 'test_2d5pt_Roofline.pickle')

        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('SandyBridgeEP_E5-2680.yml'),
                                  '-p', 'Roofline',
                                  self._find_file('2d-5pt.c'),
                                  '-D', 'N', '1024-4096:3log2',
                                  '-D', 'M', '50',
                                  '-vvv',
                                  '--store', store_file])
        kc.check_arguments(args, parser)
        kc.run(parser, args, output_file=sys.stdout)

        with open(store_file, 'rb') as f:
            results = pickle.load(f)

        # Check for correct variations of constants
        self.assertEqual(len(results), 3)

        # Check if results contains correct kernel and some other infoormation
        key = [k for k in results if ('define', (('M', 50), ('N', 4096))) in k][0]
        key_dict = dict(key)
        self.assertEqual(key_dict['pmodel'], 'Roofline')

        # Output of first result:
        result = results[key]
        assertRelativlyEqual(result['min performance']['FLOP/s'], 4720000000.0, 0.01)
        self.assertEqual(result['bottleneck level'], 1)

        expected_btlncks = [{'arithmetic intensity': 0.029411764705882353,
                             'bandwidth': PrefixedUnit(84.07, u'G', u'B/s'),
                             'bw kernel': 'load',
                             'level': u'L1',
                             'performance': PrefixedUnit(9.89, u'G', u'FLOP/s')
                             },
                            {'arithmetic intensity': 0.025,
                             'bandwidth': PrefixedUnit(47.24, u'G', u'B/s'),
                             'bw kernel': 'triad',
                             'level': u'L2',
                             'performance': PrefixedUnit(4.72, u'G', u'FLOP/s')},
                            {'arithmetic intensity': 0.041,
                             'bandwidth': PrefixedUnit(32.9, 'G', 'B/s'),
                             'bw kernel': 'copy',
                             'level': u'L3',
                             'performance': PrefixedUnit(5.33, u'G', u'FLOP/s')},
                            {'arithmetic intensity': float('inf'),
                             'bandwidth': PrefixedUnit(12.01, u'G', u'B/s'),
                             'bw kernel': 'load',
                             'level': u'MEM',
                             'performance': PrefixedUnit(float('inf'), u'', u'FLOP/s')}]

        for i, btlnck in enumerate(expected_btlncks):
            for k, v in btlnck.items():
                if type(v) is not str:
                    if k == 'performance':
                        assertRelativlyEqual(result['mem bottlenecks'][i][k]['FLOP/s'], v, 0.05)
                    else:
                        assertRelativlyEqual(result['mem bottlenecks'][i][k], v, 0.05)
                else:
                    self.assertEqual(result['mem bottlenecks'][i][k], v)

    def test_sclar_product_ECMData(self):
        store_file = os.path.join(self.temp_dir, 'test_scalar_product_ECMData.pickle')

        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('HaswellEP_E5-2695v3.yml'),
                                  '-p', 'ECMData',
                                  self._find_file('scalar_product.c'),
                                  '-D', 'N', '10000',
                                  '-vvv',
                                  '--store', store_file])
        kc.check_arguments(args, parser)
        kc.run(parser, args, output_file=sys.stdout)

        with open(store_file, 'rb') as f:
            results = pickle.load(f)

        # Output of first result:
        result = next(iter(results.values()))

        # 2 Misses in L1, since sizeof(a)+sizeof(b) = 156kB > L1
        assertRelativlyEqual(result['L2'], 2, 0.05)
        self.assertAlmostEqual(result['L3'], 0.0, places=2)
        self.assertAlmostEqual(result['MEM'], 0.0, places=2)

    def test_copy_ECMData(self):
        store_file = os.path.join(self.temp_dir, 'test_copy_ECMData.pickle')

        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('HaswellEP_E5-2695v3.yml'),
                                  '-p', 'ECMData',
                                  self._find_file('copy.c'),
                                  '-D', 'N', '1000000',
                                  '-vvv',
                                  '--unit=cy/CL',
                                  '--store', store_file])
        kc.check_arguments(args, parser)
        kc.run(parser, args, output_file=sys.stdout)

        with open(store_file, 'rb') as f:
            results = pickle.load(f)

        # Output of first result:
        result = next(iter(results.values()))

        # 2 arrays * 1000000 doubles/array * 8 Bytes/double ~ 15MB
        # -> L3
        assertRelativlyEqual(result['L2'], 3, 0.05)
        assertRelativlyEqual(result['L3'], 6, 0.05)
        self.assertAlmostEqual(result['MEM'], 0, places=2)

    def test_copy_ECMData_LC(self):
        store_file = os.path.join(self.temp_dir, 'test_copy_ECMData_LC.pickle')

        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('HaswellEP_E5-2695v3.yml'),
                                  '-p', 'ECMData',
                                  self._find_file('copy.c'),
                                  '-D', 'N', '1000000',
                                  '-vvv',
                                  '--unit=cy/CL',
                                  '--cache-predictor=LC',
                                  '--store', store_file])
        kc.check_arguments(args, parser)
        kc.run(parser, args, output_file=sys.stdout)

        with open(store_file, 'rb') as f:
            results = pickle.load(f)

        # Output of first result:
        result = next(iter(results.values()))

        # 2 arrays * 1000000 doubles/array * 8 Bytes/double ~ 15MB
        # -> L3
        assertRelativlyEqual(result['L2'], 3, 0.05)
        assertRelativlyEqual(result['L3'], 6, 0.05)
        self.assertAlmostEqual(result['MEM'], 0, places=0)

    @unittest.skipUnless(find_executable('gcc'), "GCC not available")
    def test_2d5pt_ECMCPU(self):
        store_file = os.path.join(self.temp_dir, 'test_2d5pt_ECMCPU.pickle')

        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('SandyBridgeEP_E5-2680.yml'),
                                  '-p', 'ECMCPU',
                                  self._find_file('2d-5pt.c'),
                                  '-D', 'N', '2000',
                                  '-D', 'M', '1000',
                                  '-vvv',
                                  '--unit=cy/CL',
                                  '--compiler=gcc',
                                  '--store', store_file])
        kc.check_arguments(args, parser)
        kc.run(parser, args, output_file=sys.stdout)

        with open(store_file, 'rb') as f:
            results = pickle.load(f)

        # Output of first result:
        result = next(iter(results.values()))

        assertRelativlyEqual(result['T_comp'], 11, 0.2)
        assertRelativlyEqual(result['T_RegL1'], 8, 0.2)

    @unittest.skipUnless(find_executable('gcc'), "GCC not available")
    def test_2d5pt_ECMCPU_OSACA(self):
        store_file = os.path.join(self.temp_dir, 'test_2d5pt_ECMCPU.pickle')

        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('SandyBridgeEP_E5-2680.yml'),
                                  '-p', 'ECMCPU',
                                  self._find_file('2d-5pt.c'),
                                  '-D', 'N', '2000',
                                  '-D', 'M', '1000',
                                  '-vvv',
                                  '--unit=cy/CL',
                                  '--compiler=gcc',
                                  '-i', 'OSACA',
                                  '--store', store_file])
        kc.check_arguments(args, parser)
        kc.run(parser, args, output_file=sys.stdout)

        with open(store_file, 'rb') as f:
            results = pickle.load(f)

        # Output of first result:
        result = next(iter(results.values()))

        assertRelativlyEqual(result['T_comp'], 10, 0.2)
        assertRelativlyEqual(result['T_RegL1'], 10, 0.2)


    @unittest.skipUnless(find_executable('gcc'), "GCC not available")
    def test_2d5pt_ECM(self):
        store_file = os.path.join(self.temp_dir, 'test_2d5pt_ECM.pickle')

        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('SandyBridgeEP_E5-2680.yml'),
                                  '-p', 'ECM',
                                  self._find_file('2d-5pt.c'),
                                  '-D', 'N', '2000',
                                  '-D', 'M', '1000',
                                  '-vvv',
                                  '--compiler=gcc',
                                  '--unit=cy/CL',
                                  '--store', store_file])
        kc.check_arguments(args, parser)
        kc.run(parser, args, output_file=sys.stdout)

        with open(store_file, 'rb') as f:
            results = pickle.load(f)

        # Output of first result:
        result = next(iter(results.values()))
        
        # 2 * 2000*1000 * 8 = 31MB
        # -> no full caching
        # applying layer-conditions:
        # 3 * 2000 * 8 ~ 47kB
        # -> layer-condition in L2
        assertRelativlyEqual(result['T_comp'], 11, 0.2)
        assertRelativlyEqual(result['T_RegL1'], 8, 0.2)
        assertRelativlyEqual(result['L2'], 10, 0.05)
        assertRelativlyEqual(result['L3'], 6, 0.05)
        assertRelativlyEqual(result['MEM'], 13, 0.05)

    @unittest.skipUnless(find_executable('gcc'), "GCC not available")
    def test_2d5pt_RooflineIACA(self):
        store_file = os.path.join(self.temp_dir, 'test_2d5pt_RooflineIACA.pickle')

        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('SandyBridgeEP_E5-2680.yml'),
                                  '-p', 'RooflineIACA',
                                  self._find_file('2d-5pt.c'),
                                  '-D', 'N', '4000',
                                  '-D', 'M', '1000',
                                  '-vvv',
                                  '--unit=FLOP/s',
                                  '--compiler=gcc',
                                  '--store', store_file])
        kc.check_arguments(args, parser)
        kc.run(parser, args, output_file=sys.stdout)

        with open(store_file, 'rb') as f:
            results = pickle.load(f)

        # Output of first result:
        result = next(iter(results.values()))
        
        assertRelativlyEqual(result['min performance']['FLOP/s'], 2900000000.0, 0.05)
        self.assertEqual(result['bottleneck level'], 3)

    @unittest.skipUnless(find_executable('gcc'), "GCC not available")
    @unittest.skipUnless(find_executable('likwid-perfctr'), "LIKWID not available")
    @unittest.skipIf(platform.system() == "Darwin", "Won't build on OS X.")
    def test_2d5pt_Benchmark(self):
        store_file = os.path.join(self.temp_dir, 'test_2d5pt_Benchmark.pickle')

        # patch environment to include dummy likwid
        environ_orig = os.environ
        os.environ['PATH'] = self._find_file('dummy_likwid') + ':' + os.environ['PATH']
        os.environ['LIKWID_LIB'] = ''
        os.environ['LIKWID_INCLUDE'] = '-I' + self._find_file('dummy_likwid/include')

        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('SandyBridgeEP_E5-2680.yml'),
                                  '-p', 'Benchmark',
                                  '--ignore-warnings',
                                  self._find_file('2d-5pt.c'),
                                  '-D', 'N', '1000',
                                  '-D', 'M', '1000',
                                  '--no-phenoecm',
                                  '-vvv',
                                  '--compiler=gcc',
                                  '--store', store_file])
        kc.check_arguments(args, parser)
        kc.run(parser, args, output_file=sys.stdout)

        # restore environment
        os.environ = environ_orig

        with open(store_file, 'rb') as f:
            results = pickle.load(f)

        # Output of first result:
        result = next(iter(results.values()))

        correct_results = {
            'Iterations per repetition': 996004,
            'MEM BW [MByte/s]': 272.7272,
            'MEM volume (per repetition) [B]': 252525.2,
            'Performance [MFLOP/s]': 32.27,
            'Performance [MIt/s]': 8.07,
            'Performance [MLUP/s]': 8.07,
            'Runtime (per cacheline update) [cy/CL]': 2677.34,
            'Runtime (per repetition) [s]': 0.123456}

        for k, v in correct_results.items():
            self.assertAlmostEqual(result[k], v, places=1)

    def test_2d5pt_pragma(self):

        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('SandyBridgeEP_E5-2680.yml'),
                                  '-p', 'ECMData',
                                  self._find_file('2d-5pt_pragma.c'),
                                  '-D', 'N', '1000',
                                  '-D', 'M', '50'])
        kc.check_arguments(args, parser)
        kc.run(parser, args, output_file=sys.stdout)

    @ignore_warnings
    def test_argument_parser_asm_block(self):
        # valid --asm-block
        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('SandyBridgeEP_E5-2680.yml'),
                                  '-p', 'Benchmark',
                                  self._find_file('2d-5pt.c'),
                                  '--asm-block', 'auto'])
        kc.check_arguments(args, parser)

        # valid --asm-block
        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('SandyBridgeEP_E5-2680.yml'),
                                  '-p', 'Benchmark',
                                  self._find_file('2d-5pt.c'),
                                  '--asm-block', 'manual'])
        kc.check_arguments(args, parser)

        # valid --asm-block
        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('SandyBridgeEP_E5-2680.yml'),
                                  '-p', 'Benchmark',
                                  self._find_file('2d-5pt.c'),
                                  '--asm-block', '23'])
        kc.check_arguments(args, parser)

        # invalid --asm-block
        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('SandyBridgeEP_E5-2680.yml'),
                                  '-p', 'Benchmark',
                                  self._find_file('2d-5pt.c'),
                                  '--asm-block', 'foobar'])
        with self.assertRaises(SystemExit) as cm:
            kc.check_arguments(args, parser)
        args.code_file.close()
        args.machine.close()
        self.assertEqual(cm.exception.code, 2)

    @ignore_warnings
    def test_argument_parser_define(self):
        # invalid --define
        parser = kc.create_parser()
        with self.assertRaises(SystemExit) as cm:
            args = parser.parse_args(['-m', self._find_file('SandyBridgeEP_E5-2680.yml'),
                                      '-p', 'Benchmark',
                                      self._find_file('2d-5pt.c'),
                                      '--define', 'M', '1000', '23'])
        self.assertEqual(cm.exception.code, 2)

        # invalid --define
        parser = kc.create_parser()
        with self.assertRaises(SystemExit) as cm:
            args = parser.parse_args(['-m', self._find_file('SandyBridgeEP_E5-2680.yml'),
                                      '-p', 'Benchmark',
                                      self._find_file('2d-5pt.c'),
                                      '--define', 'N'])
        self.assertEqual(cm.exception.code, 2)

        # invalid --define
        parser = kc.create_parser()
        with self.assertRaises(SystemExit) as cm:
            args = parser.parse_args(['-m', self._find_file('SandyBridgeEP_E5-2680.yml'),
                                      '-p', 'Benchmark',
                                      self._find_file('2d-5pt.c'),
                                      '--define', 'M', '1000', '23'])
        self.assertEqual(cm.exception.code, 2)

        # invalid --define
        parser = kc.create_parser()
        with self.assertRaises(SystemExit) as cm:
            args = parser.parse_args(['-m', self._find_file('SandyBridgeEP_E5-2680.yml'),
                                      '-p', 'Benchmark',
                                      self._find_file('2d-5pt.c'),
                                      '--define', 'M'])
        self.assertEqual(cm.exception.code, 2)

        # invalid --define
        parser = kc.create_parser()
        with self.assertRaises(SystemExit) as cm:
            args = parser.parse_args(['-m', self._find_file('SandyBridgeEP_E5-2680.yml'),
                                      '-p', 'Benchmark',
                                      self._find_file('2d-5pt.c'),
                                      '--define', 'M', 'foobar'])
        self.assertEqual(cm.exception.code, 2)

        # valid --define
        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('SandyBridgeEP_E5-2680.yml'),
                                  '-p', 'Benchmark',
                                  self._find_file('2d-5pt.c'),
                                  '--define', 'M', '23'])
        kc.check_arguments(args, parser)
        self.assertEqual(cm.exception.code, 2)

        # valid --define
        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('SandyBridgeEP_E5-2680.yml'),
                                  '-p', 'Benchmark',
                                  self._find_file('2d-5pt.c'),
                                  '--define', 'M', '23-42'])
        kc.check_arguments(args, parser)
        self.assertEqual(args.define[0][0], 'M')
        self.assertEqual(list(args.define[0][1]), list(range(23, 43)))

        # valid --define
        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('SandyBridgeEP_E5-2680.yml'),
                                  '-p', 'Benchmark',
                                  self._find_file('2d-5pt.c'),
                                  '--define', 'M', '23-42:4'])
        kc.check_arguments(args, parser)
        self.assertEqual(args.define[0][0], 'M')
        self.assertEqual(list(args.define[0][1]), [23, 29, 36, 42])

        # valid --define
        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('SandyBridgeEP_E5-2680.yml'),
                                  '-p', 'Benchmark',
                                  self._find_file('2d-5pt.c'),
                                  '--define', 'M', '1-8:4log2'])
        kc.check_arguments(args, parser)
        self.assertEqual(args.define[0][0], 'M')
        self.assertEqual(list(args.define[0][1]), [1, 2, 4, 8])

        # valid --define
        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('SandyBridgeEP_E5-2680.yml'),
                                  '-p', 'Benchmark',
                                  self._find_file('2d-5pt.c'),
                                  '--define', 'M', '10-1000:3log10'])
        kc.check_arguments(args, parser)
        self.assertEqual(args.define[0][0], 'M')
        self.assertEqual(list(args.define[0][1]), [10, 100, 1000])

    def test_space_linear(self):
        self.assertEqual(list(kc.space(1, 10, 10)), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertEqual(list(kc.space(1, 10, 3)), [1, 6, 10])
        self.assertEqual(list(kc.space(1, 10, 9, endpoint=False)), [1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(list(kc.space(20, 40, 2)), [20, 40])

    def test_space_log10(self):
        self.assertEqual(list(kc.space(1, 1000, 4, log=True)), [1, 10, 100, 1000])
        self.assertEqual(list(kc.space(1, 10000, 3, log=True)), [1, 100, 10000])
        self.assertEqual(list(kc.space(1, 100, 2, endpoint=False, log=True)), [1, 10])
        self.assertEqual(list(kc.space(10, 100, 2, log=True)), [10, 100])

    def test_space_log2(self):
        self.assertEqual(list(kc.space(1, 8, 4, log=True, base=2)), [1, 2, 4, 8])
        self.assertEqual(list(kc.space(1, 16, 3, log=True, base=2)), [1, 4, 16])
        self.assertEqual(list(kc.space(1, 4, 2, endpoint=False, log=True, base=2)), [1, 2])
        self.assertEqual(list(kc.space(4, 8, 2, log=True, base=2)), [4, 8])


if __name__ == '__main__':
    unittest.main(buffer=True)
