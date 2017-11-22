#!/usr/bin/env python
import unittest
import sys


suite = unittest.TestLoader().loadTestsFromNames(
    [
        'test_kerncraft',
        'test_intervals',
        'test_kernel',
        'test_layer_condition',
        'test_likwid_bench_auto'
    ]
)

testresult = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if testresult.wasSuccessful() else 1)
