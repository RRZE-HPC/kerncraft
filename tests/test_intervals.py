#!/usr/bin/env python3
"""
Unit tests for intervals module
"""
import unittest

from kerncraft.intervals import Intervals


class TestIntervals(unittest.TestCase):
    def test_init(self):
        self.assertEqual(Intervals([0, 10]).data, [[0, 10]])
        self.assertEqual(Intervals([0, 10], [1, 9]).data, [[0, 10]])
        self.assertEqual(Intervals([0, 10], [5, 15]).data, [[0, 15]])
        self.assertEqual(Intervals([-5, 5], [0, 10]).data, [[-5, 10]])
        self.assertEqual(Intervals([0, 9], [10, 11]).data, [[0, 9], [10, 11]])
        self.assertEqual(Intervals([0, 10], [10, 11]).data, [[0, 11]])

    def test_union(self):
        self.assertEqual(Intervals([0, 5]) & Intervals([1, 9]), Intervals([0, 9]))
        self.assertEqual(Intervals([0, 5]) & Intervals([5, 9]), Intervals([0, 9]))
        self.assertEqual(Intervals([2, 4]) & Intervals([0, 9]), Intervals([0, 9]))

    def test_len(self):
        self.assertEqual(len(Intervals([1, 2])), 1)
        self.assertEqual(len(Intervals([1, 3], [10, 15])), 2 + 5)

    def test_contains(self):
        self.assertTrue(0 in Intervals([0, 10]))
        self.assertTrue(5 in Intervals([0, 2], [4, 10]))
        self.assertFalse(10 in Intervals([0, 10]))
        self.assertFalse(3 in Intervals([0, 2], [4, 10]))
