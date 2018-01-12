#!/usr/bin/env python3
"""Performance model base class."""

class PerformanceModel:
    """Base class for performance models"""
    # The name of the performance model (no abreviatation)
    name = "performance-model name"

    @classmethod
    def configure_arggroup(cls, parser):
        """Configure argument parser."""
        pass

    def __init__(self):
        # Results dictionary for machine readable analysis report
        self.results = {}

    def analyze(self):
        """Analyze the kernel with regard to the machine definition and cli arguments passed."""
        raise NotImplementedError()

    def report(self):
        """Return a readable text output with analysis report."""
        raise NotImplementedError()
