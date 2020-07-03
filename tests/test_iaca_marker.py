#!/usr/bin/env python3
"""
High-level tests for the IACA marker and loop detection in incore_model.py
"""
import os
import unittest
from copy import copy

from kerncraft.incore_model import parse_asm, ISA, find_basic_loop_bodies


class TestIACAMarker(unittest.TestCase):
    @staticmethod
    def _find_file(name):
        testdir = os.path.dirname(__file__)
        name = os.path.join(testdir, 'test_files', 'iaca_marker_examples', name)
        assert os.path.exists(name)
        return name

    def tests_x86_2d5pt_constcoeffs(self):
        with open(self._find_file('2d-5pt-constcoeffs.s')) as f:
            assembly_orig = f.read()
        assembly = parse_asm(assembly_orig, 'x86')
        blocks = find_basic_loop_bodies(assembly)
        isa = ISA.get_isa('x86')
        best_block_label = isa.select_best_block(blocks)
        best_block_lines = blocks[best_block_label]

        self.assertEqual(best_block_label, '.L36')
        self.assertEqual(isa.get_pointer_increment(best_block_lines), 8)

    def tests_x86_2d5pt_varcoeffs(self):
        with open(self._find_file('2d-5pt-varcoeffs.s')) as f:
            assembly_orig = f.read()
        assembly = parse_asm(assembly_orig, 'x86')
        blocks = find_basic_loop_bodies(assembly)
        isa = ISA.get_isa('x86')
        best_block_label = isa.select_best_block(blocks)
        best_block_lines = blocks[best_block_label]

        self.assertEqual(best_block_label, '.L43')
        self.assertEqual(isa.get_pointer_increment(best_block_lines), 16)

    def tests_x86_3d25pt_semi(self):
        with open(self._find_file('3d-25pt_semi.s')) as f:
            assembly_orig = f.read()
        assembly = parse_asm(assembly_orig, 'x86')
        blocks = find_basic_loop_bodies(assembly)
        isa = ISA.get_isa('x86')
        best_block_label = isa.select_best_block(blocks)
        best_block_lines = blocks[best_block_label]

        self.assertEqual(best_block_label, 'LBB0_62')
        #self.assertEqual(isa.get_pointer_increment(best_block_lines), 8)

    def tests_x86_matvec_trans(self):
        with open(self._find_file('matvec_trans.s')) as f:
            assembly_orig = f.read()
        assembly = parse_asm(assembly_orig, 'x86')
        blocks = find_basic_loop_bodies(assembly)
        isa = ISA.get_isa('x86')
        best_block_label = isa.select_best_block(blocks)
        best_block_lines = blocks[best_block_label]

        self.assertEqual(best_block_label, 'LBB0_30')
        self.assertEqual(isa.get_pointer_increment(best_block_lines), 64)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestIACAMarker)
    unittest.TextTestRunner(verbosity=2, buffer=True).run(suite)
