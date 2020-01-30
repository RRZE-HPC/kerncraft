#!/usr/bin/env python3
"""
High-level tests for the IACA marker and loop detection in iaca.py
"""
import os
import unittest
from copy import copy
from io import StringIO

from kerncraft.incore_model import asm_instrumentation


class TestIncoreModelX86(unittest.TestCase):
    @staticmethod
    def _find_file(name):
        testdir = os.path.dirname(__file__)
        name = os.path.join(testdir, 'test_files', 'iaca_marker_examples', name)
        assert os.path.exists(name)
        return name

    def test_2d5pt_constcoeffs(self):
        with open(self._find_file('2d-5pt-constcoeffs.s')) as f:
            block_lines, pointer_increment = asm_instrumentation(f)

        self.assertEqual(block_lines[0]['label'], '.L36')
        self.assertEqual(pointer_increment, 8)

    def test_2d5pt_varcoeffs(self):
        with open(self._find_file('2d-5pt-varcoeffs.s')) as f:
            block_lines, pointer_increment = asm_instrumentation(f)

        self.assertEqual(block_lines[0]['label'], '.L43')
        self.assertEqual(pointer_increment, 16)

    def test_3d25pt_semi(self):
        with open(self._find_file('3d-25pt_semi.s')) as f:
            block_lines, pointer_increment = asm_instrumentation(f, pointer_increment=8)

        self.assertEqual(block_lines[0]['label'], 'LBB0_62')
        #self.assertEqual(pointer_increment, 8)

    def test_matvec_trans(self):
        with open(self._find_file('matvec_trans.s')) as f:
            block_lines, pointer_increment = asm_instrumentation(f)

        self.assertEqual(block_lines[0]['label'], 'LBB0_30')
        self.assertEqual(pointer_increment, 64)

    def test_increment_detection(self):
        test_cases = [
            ("""
            .L19:
                vmovupd	(%rcx), %ymm4
                vmovupd	32(%rcx), %ymm13
                vmovupd	64(%rcx), %ymm8
                vmovupd	96(%rcx), %ymm5
                subq	$-128, %rcx
                cmpq	%rcx, %r15
                jne	.L19
            """, 128)
        ]
        for code, correct_increment in test_cases:
            block_lines, pointer_increment = asm_instrumentation(StringIO(code))
            self.assertEqual(pointer_increment, correct_increment)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestIncoreModelX86)
    unittest.TextTestRunner(verbosity=2).run(suite)
