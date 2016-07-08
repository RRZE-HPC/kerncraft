'''Collection of performance models

This model combines all performance models currently supported by kerncraft. Only the performace
model class is exported, so please add new models to __all__.

The exported classes must have the following class level attributes:
  * name (str) is the name of the performance model (no abreviatation)
  * configure_subparser(parser) classmethod that configures the parser for cli usage
  * construct_from_args(kernel, machine, args) classmethod that construct the object
  * analyze() that analyses ther kernel with regard to the machine definition and args passed
  * report() return a readable text output with analysis report
  * results (dict) must be available after analyze has been called
'''
from .ecm import ECM, ECMData, ECMCPU
from .roofline import Roofline, RooflineIACA
from .benchmark import Benchmark
from .layer_condition import LC

__all__ = ['ECM', 'ECMData', 'ECMCPU', 'Roofline', 'RooflineIACA', 'Benchmark', 'LC']
