#!/usr/bin/env python3
"""Layer condition model and helper functions"""
import sys
from itertools import chain
from pprint import pprint
from collections import defaultdict
from functools import cmp_to_key

import sympy

from kerncraft.cacheprediction import LayerConditionPredictor
from .base import PerformanceModel


class LC(PerformanceModel):
    """
    Representation of the layer condition model.

    See https://rrze-hpc.github.io/layer condition/ for information about this analytical model.
    """

    name = "Layer Condition"

    @classmethod
    def configure_arggroup(cls, parser):
        """Configure arugment parser"""
        pass

    def __init__(self, kernel, machine, args=None, parser=None):
        """
        Create layer condition model from kernel and machine objects.

        *kernel* is a Kernel object
        *machine* describes the machine (cpu, cache and memory) characteristics
        *args* (optional) are the parsed arguments from the comand line
        """
        self.kernel = kernel
        self.machine = machine
        self._args = args
        self._parser = parser
        self.results = None

        if args:
            # handle CLI info
            pass

    def analyze(self):
        """Run complete analysis."""
        lcp = LayerConditionPredictor(self.kernel, self.machine, symbolic=True)
        self.results = lcp.results

    def report(self, output_file=sys.stdout):
        """Report generated model in human readable form."""
        if self._args and self._args.verbose > 2:
            pprint(self.results)
            print()

        for cache, cache_results in zip(
                self.machine.get_cachesim(self._args.cores).levels(with_mem=False),
                self.results['cache']):
            print("Layer conditions for", cache.name, "cache with", cache.size()//1024, "KB:")
            print("    {:>35} {:>7} {:>5}".format("condition", "misses", "hits"))
            for lc_condition in cache_results:
                print("    {!r:>35} {:>7} {:>5}".format(lc_condition['condition'],
                                                      lc_condition['misses'],
                                                      lc_condition['hits']))
            print()
