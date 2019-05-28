#!/usr/bin/env python3
"""
High-level tests for the LayerConditionPredictor and CacheSimulationPredictor
"""
import os
import unittest

from ruamel import yaml

from kerncraft.cacheprediction import LayerConditionPredictor, CacheSimulationPredictor
from kerncraft.machinemodel import MachineModel
from kerncraft.kernel import KernelCode



class TestLayerConditionPredictor(unittest.TestCase):
    def setUp(self):
        self.machine = MachineModel(self._find_file('SandyBridgeEP_E5-2680.yml'))

    @staticmethod
    def _find_file(name):
        testdir = os.path.dirname(__file__)
        name = os.path.join(testdir, 'test_files', name)
        assert os.path.exists(name)
        return name

    def test_non_variable_accesse(self):
        kernel = KernelCode('''
        double Y[s][n];
        double F[s][n];
        double A[s][s];
        double y[n];
        double h;
        
        for (int l = 0; l < s; ++l)
          for (int j = 0; j < n; ++j)
            Y[l][j] = A[l][0] * F[0][j] * h + y[j];
        ''', machine=self.machine)
        kernel.set_constant('s', 4)
        kernel.set_constant('n', 1000000)
        lcp = LayerConditionPredictor(kernel, self.machine)
        self.assertEqual(lcp.get_evicts(), [1, 1, 1, 0])
        self.assertEqual(lcp.get_misses(), [3, 3, 3, 0])
        self.assertEqual(lcp.get_hits(), [0, 0, 0, 3])



if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLayerConditionPredictor)
    unittest.TextTestRunner(verbosity=2).run(suite)
