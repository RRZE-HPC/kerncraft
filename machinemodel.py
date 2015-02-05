#!/usr/bin/env python
import re

import yaml

class MachineModel:
    def __init__(path_to_yaml):
        self._path = path_to_yaml
        self._data = {}
        with open(path_to_yaml, 'r') as f:
            d = f.read()
            assert "REQUIRED_INFORMATION" not in d, 'The provided yaml machine file is incomplete'
            self._data = yaml.load(d)
        
        self.__getitem__ = self._data.__getitem__
    
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, repr(self._path))