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
from distutils.spawn import find_executable
import platform

import sympy

from kerncraft import kerncraft as kc
from kerncraft.prefixedunit import PrefixedUnit


def assert_relativly_equal(actual, desired, rel_diff=0.0):
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
        output_stream = StringIO()

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
        kc.run(parser, args, output_file=output_stream)

        with open(store_file, 'rb') as f:
            results = pickle.load(f)

        # Check if results contains correct kernel
        self.assertEqual(list(results), ['2d-5pt.c'])

        # Check for correct variations of constants
        self.assertCountEqual(
            [sorted(map(str, r)) for r in results['2d-5pt.c']],
            [sorted(map(str, r)) for r in [
                ((sympy.var('M'), 50), (sympy.var('N'), 1000)), ((sympy.var('M'), 50),
                                                                 (sympy.var('N'), 10000)),
                ((sympy.var('M'), 50), (sympy.var('N'), 100000))]])

        # Output of first result:
        result = results['2d-5pt.c'][[k for k in results['2d-5pt.c']
                                      if (sympy.var('N'), 1000) in k][0]]

        self.assertCountEqual(result, ['ECMData'])

        ecmd = result['ECMData']
        # 2 arrays * 1000*50 doubles/array * 8 Bytes/double = 781kB
        # -> fully cached in L3
        assert_relativly_equal(ecmd['L2'], 6, 0.05)
        assert_relativly_equal(ecmd['L3'], 6, 0.05)
        self.assertAlmostEqual(ecmd['MEM'], 0.0, places=3)

    def test_2d5pt_ECMData_LC(self):
        store_file = os.path.join(self.temp_dir, 'test_2d5pt_ECMData.pickle')
        output_stream = StringIO()

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
        kc.run(parser, args, output_file=output_stream)

        with open(store_file, 'rb') as f:
            results = pickle.load(f)

        # Check if results contains correct kernel
        self.assertEqual(list(results), ['2d-5pt.c'])

        # Check for correct variations of constants
        self.assertCountEqual(
            [sorted(map(str, r)) for r in results['2d-5pt.c']],
            [sorted(map(str, r)) for r in [
                ((sympy.var('M'), 50), (sympy.var('N'), 1000)), ((sympy.var('M'), 50),
                                                                 (sympy.var('N'), 10000)),
                ((sympy.var('M'), 50), (sympy.var('N'), 100000))]])

        # Output of first result:
        result = results['2d-5pt.c'][
            [k for k in results['2d-5pt.c'] if (sympy.var('N'), 1000) in k][0]]

        self.assertCountEqual(result, ['ECMData'])

        ecmd = result['ECMData']
        # 2 arrays * 1000*50 doubles/array * 8 Bytes/double = 781kB
        # -> fully cached in L3
        assert_relativly_equal(ecmd['L2'], 6, 0.05)
        assert_relativly_equal(ecmd['L3'], 6, 0.05)
        self.assertAlmostEqual(ecmd['MEM'], 0.0, places=2)

    def test_2d5pt_Roofline(self):
        store_file = os.path.join(self.temp_dir, 'test_2d5pt_Roofline.pickle')
        output_stream = StringIO()

        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('SandyBridgeEP_E5-2680.yml'),
                                  '-p', 'Roofline',
                                  self._find_file('2d-5pt.c'),
                                  '-D', 'N', '1024-4096:3log2',
                                  '-D', 'M', '50',
                                  '-vvv',
                                  '--store', store_file])
        kc.check_arguments(args, parser)
        kc.run(parser, args, output_file=output_stream)

        with open(store_file, 'rb') as f:
            results = pickle.load(f)

        # Check if results contains correct kernel
        self.assertEqual(list(results), ['2d-5pt.c'])

        # Check for correct variations of constants
        self.assertCountEqual(
            [sorted(map(str, r)) for r in results['2d-5pt.c']],
            [sorted(map(str, r)) for r in [
                ((sympy.var('M'), 50), (sympy.var('N'), 1024)),
                ((sympy.var('M'), 50), (sympy.var('N'), 2048)),
                ((sympy.var('M'), 50), (sympy.var('N'), 4096))]])

        # Output of first result:
        result = results['2d-5pt.c'][
            [k for k in results['2d-5pt.c'] if (sympy.var('N'), 4096) in k][0]]

        self.assertCountEqual(result, ['Roofline'])

        roofline = result['Roofline']
        assert_relativly_equal(roofline['min performance']['FLOP/s'], 5115000000.0, 0.01)
        self.assertEqual(roofline['bottleneck level'], 1)

        expected_btlncks = [{'arithmetic intensity': 0.11764705882352941,
                             'bandwidth': PrefixedUnit(81.61, u'G', u'B/s'),
                             'bw kernel': 'triad',
                             'level': u'L1',
                             'performance': PrefixedUnit(9601176470.588236, u'', u'FLOP/s')
                             },
                            {'arithmetic intensity': 0.1,
                             'bandwidth': PrefixedUnit(51.15, u'G', u'B/s'),
                             'bw kernel': 'triad',
                             'level': u'L2',
                             'performance': PrefixedUnit(5115000000.0, u'', u'FLOP/s')},
                            {'arithmetic intensity': 1.0 / 6.0,
                             'bandwidth': PrefixedUnit(34815.0, 'M', 'B/s'),
                             'bw kernel': 'copy',
                             'level': u'L3',
                             'performance': PrefixedUnit(5802500000.0, u'', u'FLOP/s')},
                            {'arithmetic intensity': float('inf'),
                             'bandwidth': PrefixedUnit(12.01, u'G', u'B/s'),
                             'bw kernel': 'load',
                             'level': u'MEM',
                             'performance': PrefixedUnit(float('inf'), u'', u'FLOP/s')}]

        for i, btlnck in enumerate(expected_btlncks):
            for k, v in btlnck.items():
                if type(v) is not str:
                    if k == 'performance':
                        assert_relativly_equal(roofline['mem bottlenecks'][i][k]['FLOP/s'], v, 0.05)
                    else:
                        assert_relativly_equal(roofline['mem bottlenecks'][i][k], v, 0.05)
                else:
                    self.assertEqual(roofline['mem bottlenecks'][i][k], v)

    def test_sclar_product_ECMData(self):
        store_file = os.path.join(self.temp_dir, 'test_scalar_product_ECMData.pickle')
        output_stream = StringIO()

        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('HaswellEP_E5-2695v3.yml'),
                                  '-p', 'ECMData',
                                  self._find_file('scalar_product.c'),
                                  '-D', 'N', '10000',
                                  '-vvv',
                                  '--store', store_file])
        kc.check_arguments(args, parser)
        kc.run(parser, args, output_file=output_stream)

        with open(store_file, 'rb') as f:
            results = pickle.load(f)

        # Output of first result:
        ecmd = results['scalar_product.c'][((sympy.var('N'), 10000),)]['ECMData']

        # 2 Misses in L1, since sizeof(a)+sizeof(b) = 156kB > L1
        assert_relativly_equal(ecmd['L2'], 2, 0.05)
        self.assertAlmostEqual(ecmd['L3'], 0.0, places=2)
        self.assertAlmostEqual(ecmd['MEM'], 0.0, places=2)

    def test_copy_ECMData(self):
        store_file = os.path.join(self.temp_dir, 'test_copy_ECMData.pickle')
        output_stream = StringIO()

        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('HaswellEP_E5-2695v3.yml'),
                                  '-p', 'ECMData',
                                  self._find_file('copy.c'),
                                  '-D', 'N', '1000000',
                                  '-vvv',
                                  '--unit=cy/CL',
                                  '--store', store_file])
        kc.check_arguments(args, parser)
        kc.run(parser, args, output_file=output_stream)

        with open(store_file, 'rb') as f:
            results = pickle.load(f)

        # Output of first result:
        ecmd = results['copy.c'][((sympy.var('N'), 1000000),)]['ECMData']

        # 2 arrays * 1000000 doubles/array * 8 Bytes/double ~ 15MB
        # -> L3
        assert_relativly_equal(ecmd['L2'], 3, 0.05)
        assert_relativly_equal(ecmd['L3'], 6, 0.05)
        self.assertAlmostEqual(ecmd['MEM'], 0, places=2)

    def test_copy_ECMData_LC(self):
        store_file = os.path.join(self.temp_dir, 'test_copy_ECMData_LC.pickle')
        output_stream = StringIO()

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
        kc.run(parser, args, output_file=output_stream)

        with open(store_file, 'rb') as f:
            results = pickle.load(f)

        # Output of first result:
        ecmd = results['copy.c'][((sympy.var('N'), 1000000),)]['ECMData']

        # 2 arrays * 1000000 doubles/array * 8 Bytes/double ~ 15MB
        # -> L3
        assert_relativly_equal(ecmd['L2'], 3, 0.05)
        assert_relativly_equal(ecmd['L3'], 6, 0.05)
        self.assertAlmostEqual(ecmd['MEM'], 0, places=0)

    @unittest.skipUnless(find_executable('gcc'), "GCC not available")
    def test_2d5pt_ECMCPU(self):
        store_file = os.path.join(self.temp_dir, 'test_2d5pt_ECMCPU.pickle')
        output_stream = StringIO()

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
        kc.run(parser, args, output_file=output_stream)

        with open(store_file, 'rb') as f:
            results = pickle.load(f)

        # Check if results contains correct kernel
        self.assertEqual(list(results), ['2d-5pt.c'])

        # Check for correct variations of constants
        self.assertCountEqual(
            [sorted(map(str, r)) for r in results['2d-5pt.c']],
            [sorted(map(str, r)) for r in [((sympy.var('M'), 1000), (sympy.var('N'), 2000))]])

        # Output of first result:
        result = list(results['2d-5pt.c'].values())[0]

        self.assertCountEqual(result, ['ECMCPU'])

        ecmd = result['ECMCPU']
        assert_relativly_equal(ecmd['T_comp'], 11, 0.2)
        assert_relativly_equal(ecmd['T_RegL1'], 8, 0.2)

    @unittest.skipUnless(find_executable('gcc'), "GCC not available")
    def test_2d5pt_ECMCPU_OSACA(self):
        store_file = os.path.join(self.temp_dir, 'test_2d5pt_ECMCPU.pickle')
        output_stream = StringIO()

        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('SandyBridgeEP_E5-2680_OSACA.yml'),
                                  '-p', 'ECMCPU',
                                  self._find_file('2d-5pt.c'),
                                  '-D', 'N', '2000',
                                  '-D', 'M', '1000',
                                  '-vvv',
                                  '--unit=cy/CL',
                                  '--compiler=gcc',
                                  '--store', store_file])
        kc.check_arguments(args, parser)
        kc.run(parser, args, output_file=output_stream)
        print(output_stream.read())

        with open(store_file, 'rb') as f:
            results = pickle.load(f)

        # Check if results contains correct kernel
        self.assertEqual(list(results), ['2d-5pt.c'])

        # Check for correct variations of constants
        self.assertCountEqual(
            [sorted(map(str, r)) for r in results['2d-5pt.c']],
            [sorted(map(str, r)) for r in [((sympy.var('M'), 1000), (sympy.var('N'), 2000))]])

        # Output of first result:
        result = list(results['2d-5pt.c'].values())[0]

        self.assertCountEqual(result, ['ECMCPU'])

        ecmd = result['ECMCPU']
        assert_relativly_equal(ecmd['T_comp'], 10, 0.2)
        assert_relativly_equal(ecmd['T_RegL1'], 10, 0.2)


    @unittest.skipUnless(find_executable('gcc'), "GCC not available")
    def test_2d5pt_ECM(self):
        store_file = os.path.join(self.temp_dir, 'test_2d5pt_ECM.pickle')
        output_stream = StringIO()

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
        kc.run(parser, args, output_file=output_stream)

        with open(store_file, 'rb') as f:
            results = pickle.load(f)

        # Check if results contains correct kernel
        self.assertEqual(list(results), ['2d-5pt.c'])

        # Check for correct variations of constants
        self.assertCountEqual(
            [sorted(map(str, r)) for r in results['2d-5pt.c']],
            [sorted(map(str, r)) for r in [((sympy.var('M'), 1000), (sympy.var('N'), 2000))]])

        # Output of first result:
        result = list(results['2d-5pt.c'].values())[0]

        self.assertCountEqual(result, ['ECM'])

        ecmd = result['ECM']
        # 2 * 2000*1000 * 8 = 31MB
        # -> no full caching
        # applying layer-conditions:
        # 3 * 2000 * 8 ~ 47kB
        # -> layer-condition in L2
        assert_relativly_equal(ecmd['T_comp'], 11, 0.2)
        assert_relativly_equal(ecmd['T_RegL1'], 8, 0.2)
        assert_relativly_equal(ecmd['L2'], 10, 0.05)
        assert_relativly_equal(ecmd['L3'], 6, 0.05)
        assert_relativly_equal(ecmd['MEM'], 13, 0.05)

    @unittest.skipUnless(find_executable('gcc'), "GCC not available")
    def test_2d5pt_RooflineIACA(self):
        store_file = os.path.join(self.temp_dir, 'test_2d5pt_RooflineIACA.pickle')
        output_stream = StringIO()

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
        kc.run(parser, args, output_file=output_stream)

        with open(store_file, 'rb') as f:
            results = pickle.load(f)

        # Check if results contains correct kernel
        self.assertEqual(list(results), ['2d-5pt.c'])

        # Check for correct variations of constants
        self.assertCountEqual(
            [sorted(map(str, r)) for r in results['2d-5pt.c']],
            [sorted(map(str, r)) for r in [((sympy.var('M'), 1000), (sympy.var('N'), 4000))]])

        # Output of first result:
        result = list(results['2d-5pt.c'].values())[0]

        self.assertCountEqual(result, ['RooflineIACA'])

        roofline = result['RooflineIACA']
        assert_relativly_equal(roofline['min performance']['FLOP/s'], 2900000000.0, 0.05)
        self.assertEqual(roofline['bottleneck level'], 3)

    @unittest.skipUnless(find_executable('gcc'), "GCC not available")
    @unittest.skipUnless(find_executable('likwid-perfctr'), "LIKWID not available")
    @unittest.skipIf(platform.system() == "Darwin", "Won't build on OS X.")
    def test_2d5pt_Benchmark(self):
        store_file = os.path.join(self.temp_dir, 'test_2d5pt_Benchmark.pickle')
        output_stream = StringIO()

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
                                  '-vvv',
                                  '--compiler=gcc',
                                  '--store', store_file])
        kc.check_arguments(args, parser)
        kc.run(parser, args, output_file=output_stream)

        # restore environment
        os.environ = environ_orig

        with open(store_file, 'rb') as f:
            results = pickle.load(f)

        # Check if results contains correct kernel
        self.assertEqual(list(results), ['2d-5pt.c'])

        # Check for correct variations of constants
        self.assertCountEqual(
            [sorted(map(str, r)) for r in results['2d-5pt.c']],
            [sorted(map(str, r)) for r in [((sympy.var('M'), 1000), (sympy.var('N'), 1000))]])

        # Output of first result:
        result = list(results['2d-5pt.c'].values())[0]

        self.assertCountEqual(result, ['Benchmark'])

        roofline = result['Benchmark']
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
            self.assertAlmostEqual(roofline[k], v, places=1)

    def test_2d5pt_pragma(self):
        output_stream = StringIO()

        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('SandyBridgeEP_E5-2680.yml'),
                                  '-p', 'ECMData',
                                  self._find_file('2d-5pt_pragma.c'),
                                  '-D', 'N', '1000',
                                  '-D', 'M', '50'])
        kc.check_arguments(args, parser)
        kc.run(parser, args, output_file=output_stream)

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
        self.assertEqual(cm.exception.code, 2)

    def test_argument_parser_define(self):
        # invalid --define
        parser = kc.create_parser()
        with self.assertRaises(SystemExit) as cm:
            args = parser.parse_args(['-m', self._find_file('SandyBridgeEP_E5-2680.yml'),
                                      '-p', 'Benchmark',
                                      self._find_file('2d-5pt.c'),
                                      '--define', 'M', '1000', '23'])
            kc.check_arguments(args, parser)
        self.assertEqual(cm.exception.code, 2)

        # invalid --define
        parser = kc.create_parser()
        with self.assertRaises(SystemExit) as cm:
            args = parser.parse_args(['-m', self._find_file('SandyBridgeEP_E5-2680.yml'),
                                      '-p', 'Benchmark',
                                      self._find_file('2d-5pt.c'),
                                      '--define', 'N'])
            kc.check_arguments(args, parser)
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
        self.assertEqual(cm.exception.code, 2)

        # valid --define
        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('SandyBridgeEP_E5-2680.yml'),
                                  '-p', 'Benchmark',
                                  self._find_file('2d-5pt.c'),
                                  '--define', 'M', '23-42'])
        self.assertEqual(args.define[0][0], 'M')
        self.assertEqual(list(args.define[0][1]), list(range(23, 43)))

        # valid --define
        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('SandyBridgeEP_E5-2680.yml'),
                                  '-p', 'Benchmark',
                                  self._find_file('2d-5pt.c'),
                                  '--define', 'M', '23-42:4'])
        self.assertEqual(args.define[0][0], 'M')
        self.assertEqual(list(args.define[0][1]), [23, 29, 36, 42])

        # valid --define
        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('SandyBridgeEP_E5-2680.yml'),
                                  '-p', 'Benchmark',
                                  self._find_file('2d-5pt.c'),
                                  '--define', 'M', '1-8:4log2'])
        self.assertEqual(args.define[0][0], 'M')
        self.assertEqual(list(args.define[0][1]), [1, 2, 4, 8])

        # valid --define
        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('SandyBridgeEP_E5-2680.yml'),
                                  '-p', 'Benchmark',
                                  self._find_file('2d-5pt.c'),
                                  '--define', 'M', '10-1000:3log10'])
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
    unittest.main()
