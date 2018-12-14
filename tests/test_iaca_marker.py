#!/usr/bin/env python3
"""
High-level tests for the IACA marker and loop detection in iaca.py
"""
import os
import unittest
from copy import copy

from kerncraft.iaca import find_asm_blocks, select_best_block, strip_and_uncomment
from kerncraft.iaca import strip_unreferenced_labels


class TestIACAMarker(unittest.TestCase):
    @staticmethod
    def _find_file(name):
        testdir = os.path.dirname(__file__)
        name = os.path.join(testdir, 'test_files', 'iaca_marker_examples', name)
        assert os.path.exists(name)
        return name

    def tests_2d5pt_constcoeffs(self):
        with open(self._find_file('2d-5pt-constcoeffs.s')) as f:
            assembly_orig = f.readlines()
        assembly = strip_and_uncomment(copy(assembly_orig))
        assembly = strip_unreferenced_labels(assembly)
        blocks = find_asm_blocks(assembly)
        block_idx = select_best_block(blocks)
        best_block = blocks[block_idx][1]

        self.assertEqual(best_block['labels'], ['.L36'])
        self.assertEqual(best_block['pointer_increment'], 8)

    def tests_2d5pt_varcoeffs(self):
        with open(self._find_file('2d-5pt-varcoeffs.s')) as f:
            assembly_orig = f.readlines()
        assembly = strip_and_uncomment(copy(assembly_orig))
        assembly = strip_unreferenced_labels(assembly)
        blocks = find_asm_blocks(assembly)
        block_idx = select_best_block(blocks)
        best_block = blocks[block_idx][1]

        self.assertEqual(best_block['labels'], ['.L43'])
        self.assertEqual(best_block['pointer_increment'], 16)

    def tests_3d25pt_semi(self):
        with open(self._find_file('3d-25pt_semi.s')) as f:
            assembly_orig = f.readlines()
        assembly = strip_and_uncomment(copy(assembly_orig))
        assembly = strip_unreferenced_labels(assembly)
        blocks = find_asm_blocks(assembly)
        block_idx = select_best_block(blocks)
        best_block = blocks[block_idx][1]

        self.assertEqual(best_block['labels'], ['LBB0_62'])
        #self.assertEqual(best_block['pointer_increment'], 8)

    def tests_matvec_trans(self):
        with open(self._find_file('matvec_trans.s')) as f:
            assembly_orig = f.readlines()
        assembly = strip_and_uncomment(copy(assembly_orig))
        assembly = strip_unreferenced_labels(assembly)
        blocks = find_asm_blocks(assembly)
        block_idx = select_best_block(blocks)
        best_block = blocks[block_idx][1]

        self.assertEqual(best_block['labels'], ['LBB0_30'])
        self.assertEqual(best_block['pointer_increment'], 64)



if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestIACAMarker)
    unittest.TextTestRunner(verbosity=2).run(suite)
