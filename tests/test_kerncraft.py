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

import six

sys.path.insert(0, '..')
from kerncraft import kerncraft as kc


class TestKerncraft(unittest.TestCase):
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
    
    def test_2d5pt_ECMData(self):
        store_file = os.path.join(self.temp_dir, 'test_2d5pt_ECMData.pickle')
        output_stream = StringIO()
        
        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('phinally_gcc.yaml'),
                                  '-p', 'ECMData',
                                  self._find_file('2d-5pt.c'),
                                  '-D', 'N', '1000-100000:3log10',
                                  '-D', 'M', '50',
                                  '-vvv',
                                  '--store', store_file])
        kc.check_arguments(args, parser)
        kc.run(parser, args, output_file=output_stream)
        
        results = pickle.load(open(store_file, 'rb'))
        
        # Check if results contains correct kernel
        self.assertEqual(list(results), ['2d-5pt.c'])
        
        # Check for correct variations of constants
        six.assertCountEqual(self,
            [sorted(r) for r in results['2d-5pt.c']],
            [sorted(r) for r in [
                (('M', 50), ('N', 1000)), (('M', 50), ('N', 10000)), (('M', 50), ('N', 100000))]])
        
        # Output of first result:
        result = results['2d-5pt.c'][[k for k in results['2d-5pt.c'] if ('N', 1000) in k][0]]
        
        six.assertCountEqual(self, result, ['ECMData'])
        
        ecmd = result['ECMData']
        self.assertAlmostEqual(ecmd['L1-L2'], 10, places=1)
        self.assertAlmostEqual(ecmd['L2-L3'], 6, places=1)
        self.assertAlmostEqual(ecmd['L3-MEM'], 3.891891891891892, places=0)

    def test_2d5pt_Roofline(self):
        store_file = os.path.join(self.temp_dir, 'test_2d5pt_Roofline.pickle')
        output_stream = StringIO()
        
        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('phinally_gcc.yaml'),
                                  '-p', 'Roofline',
                                  self._find_file('2d-5pt.c'),
                                  '-D', 'N', '1024-4096:3log2',
                                  '-D', 'M', '50',
                                  '-vvv',
                                  '--store', store_file])
        kc.check_arguments(args, parser)
        kc.run(parser, args, output_file=output_stream)
        
        results = pickle.load(open(store_file, 'rb'))
        
        # Check if results contains correct kernel
        self.assertEqual(list(results), ['2d-5pt.c'])
        
        # Check for correct variations of constants
        six.assertCountEqual(self, 
            [sorted(r) for r in results['2d-5pt.c']],
            [sorted(r) for r in [
                (('M', 50), ('N', 1024)), (('M', 50), ('N', 2048)), (('M', 50), ('N', 4096))]])
        
        # Output of first result:
        result = results['2d-5pt.c'][[k for k in results['2d-5pt.c'] if ('N', 4096) in k][0]]
        
        six.assertCountEqual(self, result, ['Roofline'])
        
        roofline = result['Roofline']
        self.assertAlmostEqual(roofline['min performance'], 2900000000.0, places=0)
        self.assertEqual(roofline['bottleneck level'], 3)

    def test_sclar_product_ECMData(self):
        store_file = os.path.join(self.temp_dir, 'test_scalar_product_ECMData.pickle')
        output_stream = StringIO()
        
        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('hasep1.yaml'),
                                  '-p', 'ECMData',
                                  self._find_file('scalar_product.c'),
                                  '-D', 'N', '10000',
                                  '-vvv',
                                  '--store', store_file])
        kc.check_arguments(args, parser)
        kc.run(parser, args, output_file=output_stream)
        
        results = pickle.load(open(store_file, 'rb'))
        
        # Output of first result:
        ecmd = results['scalar_product.c'][(('N', 10000),)]['ECMData']
        
        self.assertAlmostEqual(ecmd['L1-L2'], 4, places=1)
        self.assertAlmostEqual(ecmd['L2-L3'], 5.54, places=1)
        self.assertAlmostEqual(ecmd['L3-MEM'], 0.0, places=0)

    def test_copy_ECMData(self):
        store_file = os.path.join(self.temp_dir, 'test_copy_ECMData.pickle')
        output_stream = StringIO()
        
        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('hasep1.yaml'),
                                  '-p', 'ECMData',
                                  self._find_file('copy.c'),
                                  '-D', 'N', '1000000',
                                  '-vvv',
                                  '--store', store_file])
        kc.check_arguments(args, parser)
        kc.run(parser, args, output_file=output_stream)
        
        results = pickle.load(open(store_file, 'rb'))
        
        # Output of first result:
        ecmd = results['copy.c'][(('N', 1000000),)]['ECMData']
        
        self.assertAlmostEqual(ecmd['L1-L2'], 6, places=1)
        self.assertAlmostEqual(ecmd['L2-L3'], 8.31, places=1)
        self.assertAlmostEqual(ecmd['L3-MEM'], 16.6, places=0)
    
    def test_2d5pt_ECMCPU(self):
        store_file = os.path.join(self.temp_dir, 'test_2d5pt_ECMCPU.pickle')
        output_stream = StringIO()
        
        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('phinally_gcc.yaml'),
                                  '-p', 'ECMCPU',
                                  self._find_file('2d-5pt.c'),
                                  '-D', 'N', '2000',
                                  '-D', 'M', '1000',
                                  '-vvv',
                                  '--unit=cy/CL',
                                  '--store', store_file])
        kc.check_arguments(args, parser)
        kc.run(parser, args, output_file=output_stream)
        
        results = pickle.load(open(store_file, 'rb'))
        
        # Check if results contains correct kernel
        self.assertEqual(list(results), ['2d-5pt.c'])
        
        # Check for correct variations of constants
        six.assertCountEqual(self, 
            [sorted(r) for r in results['2d-5pt.c']],
            [sorted(r) for r in [(('M', 1000), ('N', 2000))]])
        
        # Output of first result:
        result = list(results['2d-5pt.c'].values())[0]
        
        six.assertCountEqual(self, result, ['ECMCPU'])
        
        ecmd = result['ECMCPU']
        self.assertAlmostEqual(ecmd['T_OL'], 24.8, places=1)
        self.assertAlmostEqual(ecmd['T_nOL'], 20, places=1)

    def test_2d5pt_RooflineIACA(self):
        store_file = os.path.join(self.temp_dir, 'test_2d5pt_RooflineIACA.pickle')
        output_stream = StringIO()
        
        parser = kc.create_parser()
        args = parser.parse_args(['-m', self._find_file('phinally_gcc.yaml'),
                                  '-p', 'RooflineIACA',
                                  self._find_file('2d-5pt.c'),
                                  '-D', 'N', '4000',
                                  '-D', 'M', '1000',
                                  '-vvv',
                                  '--unit=FLOP/s',
                                  '--store', store_file])
        kc.check_arguments(args, parser)
        kc.run(parser, args, output_file=output_stream)
        
        results = pickle.load(open(store_file, 'rb'))
        
        # Check if results contains correct kernel
        self.assertEqual(list(results), ['2d-5pt.c'])
        
        # Check for correct variations of constants
        six.assertCountEqual(self, 
            [sorted(r) for r in results['2d-5pt.c']],
            [sorted(r) for r in [(('M', 1000), ('N', 4000))]])
        
        # Output of first result:
        result = list(results['2d-5pt.c'].values())[0]
        
        six.assertCountEqual(self, result, ['RooflineIACA'])
        
        roofline = result['RooflineIACA']
        self.assertAlmostEqual(roofline['min performance'], 2900000000.0, places=0)
        self.assertEqual(roofline['bottleneck level'], 2)

    def test_space_linear(self):
        self.assertEqual(list(kc.space(1, 10, 10)), [1,2,3,4,5,6,7,8,9,10])
        self.assertEqual(list(kc.space(1, 10, 3)), [1, 6, 10])
        self.assertEqual(list(kc.space(1, 10, 9, endpoint=False)), [1,2,3,4,5,6,7,8,9])
        self.assertEqual(list(kc.space(20, 40, 2)), [20, 40])
    
    
    def test_space_log10(self):
        self.assertEqual(list(kc.space(1, 1000, 4, log=True)), [1,10,100,1000])
        self.assertEqual(list(kc.space(1, 10000, 3, log=True)), [1, 100, 10000])
        self.assertEqual(list(kc.space(1, 100, 2, endpoint=False, log=True)), [1,10])
        self.assertEqual(list(kc.space(10, 100, 2, log=True)), [10, 100])
    
    
    def test_space_log2(self):
        self.assertEqual(list(kc.space(1, 8, 4, log=True, base=2)), [1,2,4,8])
        self.assertEqual(list(kc.space(1, 16, 3, log=True, base=2)), [1, 4, 16])
        self.assertEqual(list(kc.space(1, 4, 2, endpoint=False, log=True, base=2)), [1,2])
        self.assertEqual(list(kc.space(4, 8, 2, log=True, base=2)), [4, 8])
    

if __name__ == '__main__':
    unittest.main()