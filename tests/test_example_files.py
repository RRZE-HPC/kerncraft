#!/usr/bin/env python3
"""
Tests for validity of example files.
"""
import os
from glob import glob
import unittest

from kerncraft import machinemodel, kernel


class TestExampleFiles(unittest.TestCase):
    @staticmethod
    def _find_file(name):
        testdir = os.path.dirname(__file__)
        name = os.path.join(testdir, 'test_files', name)
        assert os.path.exists(name)
        return name

    def setUp(self):
        self.default_machine = machinemodel.MachineModel(self._find_file('SandyBridgeEP_E5-2680.yml'))

    def test_kernelfiles(self):
        kernel_files_glob = os.path.join(os.path.dirname(__file__), '../examples/kernels/', '*.c')
        for kernel_path in glob(kernel_files_glob):
            with self.subTest(kernel_path):
                with open(kernel_path) as kernel_code:
                    kernel.KernelCode(kernel_code.read(), self.default_machine,
                                      filename=kernel_path)

    def test_machinefiles(self):
        machine_files_glob = os.path.join(os.path.dirname(__file__), '../examples/machine-files/', '*.y*ml')
        for yml_path in glob(machine_files_glob):
            with self.subTest(yml_path=yml_path):
                machinemodel.MachineModel(yml_path)
