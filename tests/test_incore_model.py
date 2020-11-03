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

    def test_increment_detection_x86(self):
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
            """, 128),
            ("""
            .L3:
                vmovsd      (%rdi,%rax,8), %xmm0
                vmulsd      (%rsi,%rax,8), %xmm0, %xmm0
                movq        %rax, %rdx
                incq        %rax
                vaddsd      %xmm0, %xmm1, %xmm1
                vmovsd      %xmm1, s(%rip)
                cmpq        %rdx, %rcx
                jne .L3
            """, 8),
            ("""
            .L3:
                vmovsd  (%rdi,%rax,8), %xmm0
                vmulsd  (%rsi,%rax,8), %xmm0, %xmm0
                movq    %rax, %rdx
                incq    %rax
                vaddsd  %xmm0, %xmm1, %xmm1
                vmovsd  %xmm1, s(%rip)
                cmpq    %rdx, %rcx
                jne     .L3
            """, 8),
        ]
        for code, correct_increment in test_cases:
            block_lines, pointer_increment = asm_instrumentation(StringIO(code))
            self.assertEqual(pointer_increment, correct_increment)

    def test_increment_detection_aarch64(self):
        test_cases = [
            ("""
            .L5:
                ldr     s2, [x14, x3, lsl 2]
                ldr     s5, [x16, x3, lsl 2]
                ldr     s7, [x24, x3, lsl 2]
                ldr     s0, [x0, x4]
                fsub    s5, s5, s2
                ldr     s1, [x0, x8]
                ldp     s2, s4, [x11, 4]
                ldr     s3, [x13, 4]
                fsub    s5, s5, s7
                ldr     s6, [x10, 8]
                fmul    s1, s1, s2
                ldr     s7, [x11], 4
                fmul    s0, s0, s3
                ldr     s3, [x18, x3, lsl 2]
                fsub    s4, s4, s6
                ldr     s2, [x0, x7]
                ldr     s6, [x9, 8]
                fadd    s5, s5, s3
                ldr     s3, [x13, 8]
                fadd    s0, s0, s1
                ldr     s1, [x12, 8]
                fsub    s4, s4, s7
                fmul    s2, s2, s6
                ldr     s7, [x10]
                fsub    s3, s3, s1
                ldr     s6, [x1, x4]
                ldr     s1, [x13], 4
                fadd    s2, s0, s2
                fadd    s0, s4, s7
                fmul    s5, s5, s6
                ldr     s4, [x12]
                fsub    s3, s3, s1
                ldr     s6, [x1, x8]
                ldr     s7, [x1, x7]
                ldr     s1, [x2, x4]
                fmul    s0, s0, s6
                fadd    s2, s2, s5
                fadd    s3, s3, s4
                ldr     s5, [x12, 4]!
                ldr     s6, [x10, 4]!
                ldr     s4, [x2, x8]
                add     x8, x8, 4
                fadd    s2, s2, s0
                fmul    s5, s1, s5
                fmul    s3, s3, s7
                ldr     s0, [x2, x7]
                fmul    s4, s4, s6
                ldr     s7, [x9]
                ldr     s6, [x22, x4]
                add     x7, x7, 4
                fadd    s1, s2, s3
                fmul    s0, s0, s7
                fadd    s1, s1, s5
                fadd    s1, s1, s4
                fadd    s1, s1, s0
                fadd    s1, s1, s6
                str     s1, [x19]
                ldr     s0, [x17, x3, lsl 2]
                add     x3, x3, 1
                ldr     s3, [x9, 4]
                cmp     w20, w3
                ldr     s2, [x21, x4]
                fmul    s0, s1, s0
                fsub    s0, s0, s3
                fmul    s0, s0, s2
                fmul    s1, s0, s0
                str     s0, [x6]
                ldr     s3, [x15]
                ldr     s2, [x5]
                fadd    s1, s1, s3
                fmul    s0, s0, s2
                str     s1, [x15]
                ldr     s1, [x9, 4]!
                fadd    s0, s0, s1
                str     s0, [x23, x4]
                add     x4, x4, 4
                bgt     .L5
            """, 4),
            # ("""
            # .LBB0_6:                                //   Parent Loop BB0_2 Depth=1
            #    //     Parent Loop BB0_4 Depth=2
            #    // =>    This Inner Loop Header: Depth=3
            #    mov     x11, x22
            #    mov     x22, x6
            #    ldr     x6, [sp, #304]          // 8-byte Folded Reload
            #    ldr     x26, [sp, #288]         // 8-byte Folded Reload
            #    lsl     x21, x5, #2
            #    ldr     s0, [x1]
            #    add     x6, x6, x5
            #    ldr     s1, [x26, x21]
            #    ldr     s2, [x7, x21]
            #    ldr     s3, [x19, x21]
            #    ldr     s4, [x15, x21]
            #    ldr     s5, [x12, #4]!
            #    ldr     s6, [x3, w6, sxtw #2]
            #    add     w6, w6, #2              // =2
            #    ldr     s7, [x3, w6, sxtw #2]
            #    ldr     x6, [sp, #312]          // 8-byte Folded Reload
            #    ldr     s19, [x4, x21]
            #    ldr     s20, [x20, x21]
            #    ldr     s21, [x24, x21]
            #    add     x6, x6, x5
            #    ldr     s16, [x3, w6, sxtw #2]
            #    add     w6, w6, #2              // =2
            #    ldr     s17, [x3, w6, sxtw #2]
            #    mov     x6, x22
            #    mov     x22, x11
            #    ldr     x11, [sp, #280]         // 8-byte Folded Reload
            #    ldr     s22, [x8, x21]
            #    ldr     s23, [x23, x21]
            #    ldr     s24, [x16, x21]
            #    ldr     s18, [x11, x21]
            #    ldr     s25, [x9, #4]!
            #    ldr     s26, [x18, #4]!
            #    fmul    s0, s1, s0
            #    fmul    s1, s2, s3
            #    fmul    s3, s4, s5
            #    fsub    s5, s19, s20
            #    ldur    s27, [x1, #-4]
            #    fadd    s0, s0, s1
            #    fsub    s5, s5, s21
            #    fsub    s7, s7, s17
            #    fadd    s0, s0, s3
            #    fadd    s3, s5, s22
            #    fsub    s5, s7, s6
            #    ldur    s6, [x0, #-4]
            #    ldr     s2, [x27, x21]
            #    ldr     s4, [x0]
            #    fsub    s21, s25, s26
            #    ldr     s19, [x13, x21]
            #    ldr     s1, [x2, x21]
            #    fsub    s7, s21, s27
            #    fmul    s3, s18, s3
            #    fadd    s5, s5, s16
            #    ldr     s20, [x17, x21]
            #    ldur    s17, [x10, #-4]
            #    fadd    s6, s7, s6
            #    fadd    s0, s0, s3
            #    fmul    s3, s23, s5
            #    fmul    s5, s24, s6
            #    fadd    s0, s0, s3
            #    fmul    s2, s2, s4
            #    fadd    s0, s0, s5
            #    ldr     s3, [x28, x21]
            #    fmul    s1, s19, s1
            #    fadd    s0, s0, s2
            #    fmul    s4, s20, s17
            #    fadd    s0, s0, s1
            #    ldr     x11, [sp, #296]         // 8-byte Folded Reload
            #    fadd    s0, s0, s4
            #    fadd    s0, s3, s0
            #    str     s0, [x6, :lo12:s0]
            #    ldr     s1, [x11, x21]
            #    ldr     s2, [x10]
            #    ldr     s3, [x29, x21]
            #    add     x5, x5, #1              // =1
            #    fmul    s0, s0, s1
            #    ldr     s1, [x14, :lo12:gosa]
            #    fsub    s0, s0, s2
            #    fmul    s0, s0, s3
            #    fmul    s3, s0, s0
            #    fadd    s1, s1, s3
            #    ldr     s2, [x25, :lo12:omega]
            #    str     s0, [x22, :lo12:ss]
            #    str     s1, [x14, :lo12:gosa]
            #    ldr     s1, [x10]
            #    ldr     x10, [sp, #272]         // 8-byte Folded Reload
            #    fmul    s0, s0, s2
            #    mov     x1, x9
            #    fadd    s0, s1, s0
            #    cmp     x10, x5
            #    mov     x10, x12
            #    mov     x0, x18
            #    str     s0, [x30, x21]
            #    b.ne    .LBB0_6
            # """, 4),
            # ("""
            # .LBB0_10:                               //   Parent Loop BB0_2 Depth=1
            #    //     Parent Loop BB0_4 Depth=2
            #    // =>    This Inner Loop Header: Depth=3
            #    ldr     x8, [sp, #520]          // 8-byte Folded Reload
            #    add     x24, x8, x15
            #    mov     x13, x5
            #    ldr     x8, [sp, #568]          // 8-byte Folded Reload
            #    add     x10, x8, x15
            #    add     x7, x25, x15
            #    ldr     x8, [sp, #624]          // 8-byte Folded Reload
            #    add     x5, x8, x15
            #    add     x19, x22, x15
            #    ldr     x8, [sp, #592]          // 8-byte Folded Reload
            #    add     x21, x8, x15
            #    add     x2, x2, #2              // =2
            #    ldr     x8, [sp, #616]          // 8-byte Folded Reload
            #    add     x3, x8, x15
            #    ldr     x8, [sp, #664]          // 8-byte Folded Reload
            #    add     x14, x8, x15
            #    ldr     x8, [sp, #640]          // 8-byte Folded Reload
            #    add     x11, x8, x15
            #    ldr     x8, [sp, #632]          // 8-byte Folded Reload
            #    add     x16, x8, x15
            #    ldr     x8, [sp, #608]          // 8-byte Folded Reload
            #    ldr     x9, [sp, #504]          // 8-byte Folded Reload
            #    add     x20, x9, x15
            #    add     x28, x8, x15
            #    ldr     x8, [sp, #576]          // 8-byte Folded Reload
            #    ldr     x9, [sp, #544]          // 8-byte Folded Reload
            #    fsub    s2, s2, s3
            #    add     x23, x9, x15
            #    add     x29, x8, x15
            #    ldr     x9, [sp, #648]          // 8-byte Folded Reload
            #    ldr     s19, [x16]
            #    ldr     s22, [x4]
            #    fmul    s0, s0, s7
            #    fmul    s1, s1, s17
            #    ldr     s3, [x21]
            #    ldr     s4, [x3]
            #    ldr     s18, [x18, #8]
            #    ldr     s24, [x29, #8]
            #    ldr     s7, [x10, #8]
            #    ldr     x8, [sp, #536]          // 8-byte Folded Reload
            #    add     x30, x9, x15
            #    add     x6, x8, x15
            #    fsub    s3, s3, s22
            #    fsub    s2, s2, s19
            #    ldr     x8, [sp, #528]          // 8-byte Folded Reload
            #    ldr     s20, [x30]
            #    fmul    s4, s4, s18
            #    fsub    s7, s7, s24
            #    fadd    s0, s0, s1
            #    add     x8, x8, x15
            #    fsub    s1, s3, s16
            #    fadd    s2, s2, s20
            #    ldr     s5, [x8]
            #    fsub    s3, s7, s6
            #    fadd    s0, s0, s4
            #    ldp     s4, s6, [x4, #-8]
            #    fadd    s1, s1, s4
            #    fmul    s2, s5, s2
            #    ldr     s21, [x28]
            #    ldp     s4, s5, [x29]
            #    fadd    s3, s3, s4
            #    fadd    s0, s0, s2
            #    fmul    s1, s21, s1
            #    ldr     s23, [x7]
            #    ldr     s25, [x6]
            #    ldr     s26, [x19]
            #    ldr     s27, [x20]
            #    ldr     s2, [x18]
            #    fadd    s0, s0, s1
            #    fmul    s1, s23, s3
            #    ldr     x9, [sp, #512]          // 8-byte Folded Reload
            #    add     x9, x9, x15
            #    ldr     x26, [sp, #552]         // 8-byte Folded Reload
            #    add     x26, x26, x15
            #    fadd    s0, s0, s1
            #    fmul    s1, s25, s5
            #    ldr     x27, [sp, #560]         // 8-byte Folded Reload
            #    add     x27, x27, x15
            #    add     x15, x15, #8            // =8
            #    fadd    s0, s0, s1
            #    fmul    s1, s26, s6
            #    fadd    s0, s0, s1
            #    fmul    s1, s27, s2
            #    ldr     s2, [x0, :lo12:gosa]
            #    fadd    s0, s0, s1
            #    ldr     s1, [x23]
            #    fadd    s0, s1, s0
            #    str     s0, [x1, :lo12:s0]
            #    ldr     s1, [x9]
            #    fmul    s0, s0, s1
            #    ldr     s1, [x18, #4]
            #    fsub    s0, s0, s1
            #    ldr     s1, [x26]
            #    fmul    s0, s0, s1
            #    fmul    s1, s0, s0
            #    str     s0, [x12, :lo12:ss]
            #    fadd    s1, s2, s1
            #    ldr     s2, [x17, :lo12:omega]
            #    fmul    s2, s0, s2
            #    str     s1, [x0, :lo12:gosa]
            #    ldr     s0, [x18, #4]
            #    fadd    s0, s0, s2
            #    str     s0, [x27]
            #    ldr     s1, [x14, #4]
            #    ldr     s2, [x11, #4]
            #    ldr     s3, [x24, #4]
            #    ldr     s4, [x5, #4]
            #    fsub    s1, s2, s1
            #    ldr     s6, [x16, #4]
            #    ldp     s17, s18, [x21]
            #    ldp     s20, s21, [x4]
            #    ldp     s24, s25, [x10, #4]
            #    fmul    s3, s3, s25
            #    fmul    s4, s4, s17
            #    fsub    s2, s18, s21
            #    ldr     s26, [x10, #12]
            #    ldr     s27, [x6, #4]
            #    fsub    s1, s1, s6
            #    ldr     s5, [x3, #4]
            #    ldr     s7, [x30, #4]
            #    ldp     s30, s28, [x29, #8]
            #    ldr     s10, [x18, #12]
            #    ldur    s19, [x21, #-4]
            #    fmul    s5, s5, s10
            #    fsub    s6, s26, s28
            #    fadd    s3, s3, s4
            #    fsub    s2, s2, s19
            #    fadd    s1, s1, s7
            #    ldr     s0, [x8, #4]
            #    ldur    s22, [x4, #-4]
            #    fsub    s4, s6, s24
            #    fadd    s3, s3, s5
            #    fadd    s2, s2, s22
            #    ldr     s16, [x28, #4]
            #    fmul    s0, s0, s1
            #    ldr     s29, [x29, #4]
            #    fadd    s1, s4, s29
            #    fmul    s2, s16, s2
            #    ldr     s23, [x7, #4]
            #    fadd    s0, s3, s0
            #    fmul    s1, s23, s1
            #    fmul    s17, s27, s30
            #    ldr     s31, [x19, #4]
            #    ldr     s8, [x20, #4]
            #    ldr     s9, [x23, #4]
            #    mov     x5, x13
            #    fadd    s0, s0, s2
            #    ldr     s2, [x18, #4]
            #    fmul    s2, s8, s2
            #    fadd    s0, s0, s1
            #    fmul    s1, s31, s20
            #    fadd    s0, s0, s17
            #    fadd    s0, s0, s1
            #    fadd    s0, s0, s2
            #    fadd    s0, s9, s0
            #    str     s0, [x1, :lo12:s0]
            #    ldr     s1, [x9, #4]
            #    ldr     s2, [x26, #4]
            #    fmul    s0, s0, s1
            #    ldr     s1, [x18, #8]
            #    fsub    s0, s0, s1
            #    fmul    s0, s0, s2
            #    fmul    s1, s0, s0
            #    ldr     s2, [x0, :lo12:gosa]
            #    str     s0, [x12, :lo12:ss]
            #    fadd    s1, s2, s1
            #    ldr     s2, [x17, :lo12:omega]
            #    fmul    s0, s0, s2
            #    str     s1, [x0, :lo12:gosa]
            #    ldr     s1, [x18, #8]
            #    fadd    s0, s1, s0
            #    str     s0, [x27, #4]
            #    cmp     x13, x2
            #    b.ne    .LBB0_10
            # """, 8),
            # ("""
            # .LBB0_7:                                //   Parent Loop BB0_2 Depth=1
            #    //     Parent Loop BB0_4 Depth=2
            #    // =>    This Inner Loop Header: Depth=3
            #    str     x13, [sp, #632]         // 8-byte Folded Spill
            #    str     x18, [sp, #600]         // 8-byte Folded Spill
            #    add     x8, x4, x30
            #    add     x18, x20, x30
            #    add     x13, x13, x30
            #    str     x10, [sp, #616]         // 8-byte Folded Spill
            #    add     x10, x10, x30
            #    str     x17, [sp, #528]         // 8-byte Folded Spill
            #    str     x21, [sp, #536]         // 8-byte Folded Spill
            #    str     x23, [sp, #544]         // 8-byte Folded Spill
            #    str     x6, [sp, #552]          // 8-byte Folded Spill
            #    str     x4, [sp, #560]          // 8-byte Folded Spill
            #    str     x16, [sp, #568]         // 8-byte Folded Spill
            #    str     x3, [sp, #576]          // 8-byte Folded Spill
            #    str     x15, [sp, #584]         // 8-byte Folded Spill
            #    str     x11, [sp, #592]         // 8-byte Folded Spill
            #    stp     x8, x18, [sp, #504]     // 16-byte Folded Spill
            #    str     x13, [sp, #520]         // 8-byte Folded Spill
            #    ldp     d1, d2, [x10, #-40]
            #    ldur    d3, [x10, #-48]
            #    ldur    d5, [x13, #-8]
            #    fadd    d2, d2, d3
            #    mov     x13, x20
            #    add     x20, x7, x30
            #    ldr     d0, [x0, :lo12:c0]
            #    ldr     d17, [x2, :lo12:c1]
            #    ldur    d19, [x20, #-8]
            #    ldur    d20, [x8, #-8]
            #    add     x4, x3, x30
            #    add     x3, x6, x30
            #    mov     x6, x7
            #    mov     x7, x0
            #    mov     x0, x19
            #    add     x19, x23, x30
            #    fmul    d0, d0, d1
            #    fadd    d19, d19, d20
            #    fmul    d2, d17, d2
            #    ldur    d4, [x18, #-8]
            #    ldur    d1, [x19, #-8]
            #    fadd    d1, d4, d1
            #    fmul    d19, d17, d19
            #    fadd    d0, d0, d2
            #    ldur    d16, [x10, #-24]
            #    ldur    d3, [x10, #-56]
            #    fadd    d3, d16, d3
            #    fmul    d1, d17, d1
            #    fadd    d0, d0, d19
            #    adrp    x23, c2
            #    add     x24, x21, x30
            #    ldur    d25, [x4, #-8]
            #    ldr     d2, [x23, :lo12:c2]
            #    add     x2, x17, x30
            #    add     x25, x15, x30
            #    fadd    d4, d5, d25
            #    fmul    d3, d2, d3
            #    fadd    d0, d0, d1
            #    ldur    d6, [x24, #-8]
            #    ldur    d26, [x2, #-8]
            #    fadd    d5, d6, d26
            #    fmul    d1, d2, d4
            #    fadd    d0, d0, d3
            #    ldp     d27, d22, [x10, #-16]
            #    ldp     d28, d29, [x10, #-72]
            #    add     x22, x11, x30
            #    add     x27, x5, x30
            #    fadd    d6, d27, d29
            #    fmul    d2, d2, d5
            #    fadd    d0, d0, d1
            #    ldur    d7, [x25, #-8]
            #    ldur    d18, [x22, #-8]
            #    ldr     d1, [x29, :lo12:c3]
            #    add     x21, x1, x30
            #    add     x28, x16, x30
            #    fadd    d7, d18, d7
            #    fmul    d3, d1, d6
            #    fadd    d0, d0, d2
            #    ldur    d21, [x21, #-8]
            #    ldur    d30, [x27, #-8]
            #    fadd    d16, d21, d30
            #    fmul    d2, d1, d7
            #    fadd    d0, d0, d3
            #    fadd    d18, d22, d28
            #    fmul    d1, d1, d16
            #    fadd    d0, d0, d2
            #    mov     x17, x4
            #    adrp    x4, c4
            #    ldur    d23, [x28, #-8]
            #    ldur    d31, [x3, #-8]
            #    ldr     d2, [x4, :lo12:c4]
            #    add     x11, x12, x30
            #    add     x18, x14, x30
            #    fadd    d20, d23, d31
            #    fmul    d3, d2, d18
            #    fadd    d0, d0, d1
            #    ldur    d24, [x11, #-8]
            #    ldur    d8, [x18, #-8]
            #    fadd    d21, d24, d8
            #    fmul    d1, d2, d20
            #    fadd    d0, d0, d3
            #    fmul    d2, d2, d21
            #    fadd    d0, d0, d1
            #    adrp    x16, lap
            #    add     x26, x9, x30
            #    fadd    d0, d0, d2
            #    ldr     x8, [sp, #608]          // 8-byte Folded Reload
            #    add     x15, x8, x30
            #    add     x9, x9, #16             // =16
            #    add     x14, x14, #16           // =16
            #    add     x12, x12, #16           // =16
            #    add     x5, x5, #16             // =16
            #    add     x1, x1, #16             // =16
            #    add     x8, x8, #16             // =16
            #    str     d0, [x16, :lo12:lap]
            #    ldur    d1, [x10, #-40]
            #    fadd    d1, d1, d1
            #    ldr     d2, [x26]
            #    ldr     d3, [x15]
            #    fmul    d0, d0, d3
            #    fsub    d1, d1, d2
            #    fadd    d0, d1, d0
            #    str     d0, [x26]
            #    ldr     d0, [x20]
            #    mov     x20, x13
            #    add     x20, x20, #16           // =16
            #    ldr     x13, [sp, #504]         // 8-byte Folded Reload
            #    ldr     d1, [x13]
            #    ldr     x13, [sp, #512]         // 8-byte Folded Reload
            #    ldr     d3, [x19]
            #    mov     x19, x0
            #    mov     x0, x7
            #    ldp     d26, d27, [x10, #-32]
            #    ldp     d28, d29, [x10, #-48]
            #    ldr     d24, [x0, :lo12:c0]
            #    fmul    d24, d24, d26
            #    ldr     d2, [x13]
            #    ldr     x13, [sp, #520]         // 8-byte Folded Reload
            #    ldr     x13, [sp, #520]         // 8-byte Folded Reload
            #    fadd    d26, d27, d29
            #    ldr     d7, [x2]
            #    adrp    x2, c1
            #    mov     x7, x6
            #    ldp     d27, d30, [x10, #-8]
            #    ldr     d4, [x13]
            #    ldr     d5, [x17]
            #    ldr     d6, [x24]
            #    ldr     d18, [x21]
            #    ldr     d19, [x27]
            #    ldr     d25, [x2, :lo12:c1]
            #    fadd    d0, d0, d1
            #    fadd    d1, d2, d3
            #    fmul    d0, d25, d0
            #    fadd    d3, d4, d5
            #    fadd    d4, d6, d7
            #    fadd    d7, d18, d19
            #    fmul    d19, d25, d26
            #    ldur    d31, [x10, #-16]
            #    fadd    d2, d31, d28
            #    fmul    d1, d25, d1
            #    ldp     d29, d8, [x10, #-64]
            #    fadd    d19, d24, d19
            #    fadd    d5, d27, d8
            #    ldr     d16, [x22]
            #    ldr     d17, [x25]
            #    fadd    d6, d16, d17
            #    fadd    d0, d19, d0
            #    ldr     d19, [x23, :lo12:c2]
            #    ldr     x23, [sp, #544]         // 8-byte Folded Reload
            #    fmul    d2, d19, d2
            #    fadd    d16, d30, d29
            #    ldr     d20, [x28]
            #    ldr     d21, [x3]
            #    fadd    d0, d0, d1
            #    fmul    d1, d19, d3
            #    fadd    d17, d20, d21
            #    ldr     d22, [x11]
            #    ldr     d23, [x18]
            #    fadd    d18, d22, d23
            #    fadd    d0, d0, d2
            #    fmul    d2, d19, d4
            #    ldr     x6, [sp, #552]          // 8-byte Folded Reload
            #    ldr     x13, [sp, #632]         // 8-byte Folded Reload
            #    ldr     x17, [sp, #528]         // 8-byte Folded Reload
            #    ldr     x21, [sp, #536]         // 8-byte Folded Reload
            #    fadd    d0, d0, d1
            #    ldr     d1, [x29, :lo12:c3]
            #    fmul    d3, d1, d5
            #    ldr     x3, [sp, #576]          // 8-byte Folded Reload
            #    ldr     x11, [sp, #592]         // 8-byte Folded Reload
            #    ldr     x18, [sp, #600]         // 8-byte Folded Reload
            #    fadd    d0, d0, d2
            #    fmul    d2, d1, d6
            #    fadd    d0, d0, d3
            #    fmul    d1, d1, d7
            #    add     x18, x18, #2            // =2
            #    add     x17, x17, #16           // =16
            #    fadd    d0, d0, d2
            #    ldr     d2, [x4, :lo12:c4]
            #    ldr     x4, [sp, #560]          // 8-byte Folded Reload
            #    fmul    d3, d2, d16
            #    add     x21, x21, #16           // =16
            #    add     x23, x23, #16           // =16
            #    fadd    d0, d0, d1
            #    fmul    d1, d2, d17
            #    fadd    d0, d0, d3
            #    fmul    d2, d2, d18
            #    add     x6, x6, #16             // =16
            #    add     x7, x7, #16             // =16
            #    fadd    d0, d0, d1
            #    add     x4, x4, #16             // =16
            #    add     x13, x13, #16           // =16
            #    fadd    d0, d0, d2
            #    add     x3, x3, #16             // =16
            #    add     x11, x11, #16           // =16
            #    str     x8, [sp, #608]          // 8-byte Folded Spill
            #    str     d0, [x16, :lo12:lap]
            #    ldur    d1, [x10, #-32]
            #    ldr     x10, [sp, #616]         // 8-byte Folded Reload
            #    fadd    d1, d1, d1
            #    ldr     d2, [x15, #8]
            #    ldr     d3, [x26, #8]
            #    fmul    d0, d0, d2
            #    fsub    d1, d1, d3
            #    ldr     x16, [sp, #568]         // 8-byte Folded Reload
            #    ldr     x15, [sp, #584]         // 8-byte Folded Reload
            #    add     x16, x16, #16           // =16
            #    add     x10, x10, #16           // =16
            #    fadd    d0, d1, d0
            #    add     x15, x15, #16           // =16
            #    str     d0, [x26, #8]
            #    cmp     x19, x18
            #    b.ne    .LBB0_7
            # """, 16),
            ("""
            .LBB0_2:                                // =>This Inner Loop Header: Depth=1
                ldr     d1, [x0], #8
                subs    x9, x9, #1              // =1
                fadd    d0, d1, d0
                str     d0, [x8, :lo12:s]
                b.ne    .LBB0_2
            """, 8),
            ("""
            .L5:
                ldr     s2, [x14, x3, lsl 2]
                ldr     s5, [x16, x3, lsl 2]
                ldr     s7, [x24, x3, lsl 2]
                ldr     s0, [x0, x4]
                fsub    s5, s5, s2
                ldr     s1, [x0, x8]
                ldp     s2, s4, [x11, 4]
                ldr     s3, [x13, 4]
                fsub    s5, s5, s7
                ldr     s6, [x10, 8]
                fmul    s1, s1, s2
                ldr     s7, [x11], 4
                fmul    s0, s0, s3
                ldr     s3, [x18, x3, lsl 2]
                fsub    s4, s4, s6
                ldr     s2, [x0, x7]
                ldr     s6, [x9, 8]
                fadd    s5, s5, s3
                ldr     s3, [x13, 8]
                fadd    s0, s0, s1
                ldr     s1, [x12, 8]
                fsub    s4, s4, s7
                fmul    s2, s2, s6
                ldr     s7, [x10]
                fsub    s3, s3, s1
                ldr     s6, [x1, x4]
                ldr     s1, [x13], 4
                fadd    s2, s0, s2
                fadd    s0, s4, s7
                fmul    s5, s5, s6
                ldr     s4, [x12]
                fsub    s3, s3, s1
                ldr     s6, [x1, x8]
                ldr     s7, [x1, x7]
                ldr     s1, [x2, x4]
                fmul    s0, s0, s6
                fadd    s2, s2, s5
                fadd    s3, s3, s4
                ldr     s5, [x12, 4]!
                ldr     s6, [x10, 4]!
                ldr     s4, [x2, x8]
                add     x8, x8, 4
                fadd    s2, s2, s0
                fmul    s5, s1, s5
                fmul    s3, s3, s7
                ldr     s0, [x2, x7]
                fmul    s4, s4, s6
                ldr     s7, [x9]
                ldr     s6, [x22, x4]
                add     x7, x7, 4
                fadd    s1, s2, s3
                fmul    s0, s0, s7
                fadd    s1, s1, s5
                fadd    s1, s1, s4
                fadd    s1, s1, s0
                fadd    s1, s1, s6
                str     s1, [x19]
                ldr     s0, [x17, x3, lsl 2]
                add     x3, x3, 1
                ldr     s3, [x9, 4]
                cmp     w20, w3
                ldr     s2, [x21, x4]
                fmul    s0, s1, s0
                fsub    s0, s0, s3
                fmul    s0, s0, s2
                fmul    s1, s0, s0
                str     s0, [x6]
                ldr     s3, [x15]
                ldr     s2, [x5]
                fadd    s1, s1, s3
                fmul    s0, s0, s2
                str     s1, [x15]
                ldr     s1, [x9, 4]!
                fadd    s0, s0, s1
                str     s0, [x23, x4]
                add     x4, x4, 4
                bgt     .L5
            """, 4),
            ("""
            .L4:
                ldp     d20, d0, [x1, 16]
                ldp     d3, d2, [x1, 32]
                ldr     d1, [x19, x0, lsl 3]
                ldr     d6, [x21, x0, lsl 3]
                fadd    d2, d2, d0
                ldr     d5, [x18, x0, lsl 3]
                ldr     d0, [x5, x0, lsl 3]
                fadd    d6, d6, d1
                fmul    d1, d3, d19
                ldp     d4, d21, [x1, 48]
                fmul    d2, d2, d18
                fadd    d5, d5, d0
                ldr     d23, [x9, x0, lsl 3]
                fmul    d6, d6, d18
                ldr     d0, [x16, x0, lsl 3]
                fadd    d4, d4, d20
                ldr     d22, [x17, x0, lsl 3]
                fadd    d2, d2, d1
                ldr     d1, [x7, x0, lsl 3]
                fmul    d5, d5, d18
                ldr     d20, [x1, 8]
                fadd    d22, d22, d0
                ldr     d0, [x15, x0, lsl 3]
                fadd    d1, d23, d1
                fmul    d4, d4, d17
                fadd    d2, d2, d6
                fadd    d21, d21, d20
                ldr     d6, [x14, x0, lsl 3]
                fadd    d3, d3, d3
                fmul    d22, d22, d17
                ldr     d20, [x8, x0, lsl 3]
                fmul    d23, d1, d17
                ldr     d24, [x10, x0, lsl 3]
                fadd    d1, d2, d5
                fadd    d0, d0, d6
                fmul    d21, d21, d16
                ldr     d5, [x11, x0, lsl 3]
                ldr     d6, [x1, 64]
                ldr     d2, [x1], 8
                fadd    d1, d1, d4
                fadd    d20, d20, d5
                fmul    d0, d0, d16
                ldr     d5, [x13, x0, lsl 3]
                fadd    d6, d6, d2
                ldr     d2, [x3, x0, lsl 3]
                ldr     d4, [x12, x0, lsl 3]
                fadd    d1, d1, d22
                fmul    d20, d20, d16
                fadd    d5, d5, d2
                ldr     d2, [x2, x0, lsl 3]
                fmul    d6, d6, d7
                fadd    d4, d4, d24
                ldr     d22, [x6, x0, lsl 3]
                fadd    d1, d1, d23
                fsub    d2, d3, d2
                fmul    d5, d5, d7
                fmul    d3, d4, d7
                fadd    d1, d1, d21
                fadd    d0, d1, d0
                fadd    d0, d0, d20
                fadd    d0, d0, d6
                fadd    d0, d0, d5
                fadd    d0, d0, d3
                fmul    d1, d0, d22
                fadd    d1, d2, d1
                str     d1, [x2, x0, lsl 3]
                add     x0, x0, 1
                cmp     w20, w0
                bgt     .L4
            """, 8),
            ("""
                .LBB0_7:                                //   Parent Loop BB0_9 Depth=1
                // =>  This Inner Loop Header: Depth=2
                add x8, x24, x17
                ldur        q3, [x8, #8]
                ldp q4, q5, [x8]
                ldur        q2, [x24, #8]
                add x6, x24, x18
                ldur        q6, [x6, #8]
                fadd        v3.2d, v3.2d, v4.2d
                fadd        v3.2d, v3.2d, v5.2d
                fadd        v2.2d, v3.2d, v2.2d
                fadd        v2.2d, v2.2d, v6.2d
                fmul        v2.2d, v2.2d, v1.2d
                subs        x25, x25, #2            // =2
                str q2, [x26], #16
                add x24, x24, #16           // =16
                b.ne        .LBB0_7
            """, 16),
            ("""
                .LBB0_8:                                //   Parent Loop BB0_10 Depth=1 
                // =>  This Inner Loop Header: Depth=2
                add     x13, x19, x25
                add     x18, x0, x25
                ldur    q0, [x13, #8]
                ldr     q1, [x18]
                ldr     q2, [x13]
                add     x3, x16, x25
                ldr     q3, [x3]
                fadd    v0.2d, v0.2d, v1.2d
                ldur    q1, [x18, #8]
                fadd    v0.2d, v0.2d, v2.2d
                ldur    q2, [x3, #8]
                fadd    v0.2d, v0.2d, v3.2d
                fadd    v0.2d, v0.2d, v1.2d
                ldr     q1, [x18, #16]
                fadd    v0.2d, v0.2d, v2.2d
                ldr     q2, [x13, #16]
                add     x13, x7, x25
                fadd    v0.2d, v0.2d, v1.2d
                ldr     q1, [x3, #16]
                fadd    v0.2d, v0.2d, v2.2d
                ldur    q2, [x13, #8]
                add     x13, x6, x25
                fadd    v0.2d, v0.2d, v1.2d
                subs    x5, x5, #2              // =2
                fmul    v0.2d, v2.2d, v0.2d
                add     x25, x25, #16           // =16
                stur    q0, [x13, #8]
                b.ne    .LBB0_8
            """, 16),
        ]
        for code, correct_increment in test_cases:
            try:
                block_lines, pointer_increment = asm_instrumentation(
                    StringIO(code), isa='aarch64', pointer_increment='auto')
            except RuntimeError:
                pointer_increment = None
                print(code)
            self.assertEqual(pointer_increment, correct_increment,
                             msg='\n'.join(code.split('\n')[:10]))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestIncoreModelX86)
    unittest.TextTestRunner(verbosity=2, buffer=True).run(suite)
