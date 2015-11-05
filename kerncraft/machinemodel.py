from __future__ import absolute_import
#!/usr/bin/env python

import yaml


class MachineModel(object):
    def __init__(self, path_to_yaml):
        self._path = path_to_yaml
        self._data = {}
        with open(path_to_yaml, 'r') as f:
            self._data = yaml.load(f)

    def __getitem__(self, index):
        return self._data[index]

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, repr(self._path))
