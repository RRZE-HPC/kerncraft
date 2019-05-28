#!/usr/bin/env python3
"""
Tests for validity of example files.
"""
import os
from glob import glob
import unittest

from kerncraft import machinemodel
from kerncraft.prefixedunit import PrefixedUnit


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
