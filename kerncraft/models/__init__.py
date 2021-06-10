"""
Collection of performance models.

This model combines all performance models currently supported by kerncraft. Only the performace
model class is exported, so please add new models to __all__.
"""
from .ecm import ECM, ECMData, ECMCPU
from .roofline import RooflineFLOP, RooflineASM
from .benchmark import Benchmark
from .layer_condition import LC
from .base import PerformanceModel

RooflineIACA = RooflineASM  # for downward compatability

__all__ = ['ECM', 'ECMData', 'ECMCPU', 'RooflineFLOP', 'RooflineASM', 'Benchmark', 'LC',
           'PerformanceModel', 'RooflineIACA']
