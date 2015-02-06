#!/usr/bin/env python
import re

import yaml
import prefixedunit

class MachineModel:
    def __init__(self, path_to_yaml):
        self._path = path_to_yaml
        self._data = {}
        with open(path_to_yaml, 'r') as f:
            self._data = yaml.load(f)
        
        self.__getitem__ = self._data.__getitem__
    
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, repr(self._path))
