#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division

from ruamel import yaml
import cachesim


class MachineModel(object):
    def __init__(self, path_to_yaml=None, machine_yaml=None):
        if not path_to_yaml and not machine_yaml:
            raise ValueError('Either path_to_yaml ot machine_yaml is required')
        if path_to_yaml and machine_yaml:
            raise ValueError('Only one of path_to_yaml and machine_yaml is allowed')
        self._path = path_to_yaml
        self._data = machine_yaml
        if path_to_yaml:
            with open(path_to_yaml, 'r') as f:
                self._data = yaml.load(f)

    def __getitem__(self, index):
        return self._data[index]

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            repr(self._path or self._data['model name']),
        )

    def get_cachesim(self, cores=1):
        '''Returns a cachesim.CacheSimulator object based on the machine description
        and used core count'''
        cache_stack = []
        cache = None
        cl_size = int(self['cacheline size'])

        cs, caches, mem = cachesim.CacheSimulator.from_dict(
            {c['level']: c['cache per group']
             for c in self['memory hierarchy']
             if 'cache per group' in c})

        return cs

    def get_bandwidth(self, cache_level, read_streams, write_streams, threads_per_core, cores=None):
        '''Returns best fitting bandwidth according to parameters

        :param cores: if not given, will choose maximum bandwidth
        '''
        # try to find best fitting kernel (closest to stream seen stream counts):
        # write allocate has to be handled in kernel information (all writes are also reads)
        # TODO support for non-write-allocate architectures
        measurement_kernel = 'load'
        measurement_kernel_info = self['benchmarks']['kernels'][measurement_kernel]
        for kernel_name, kernel_info in sorted(
                self['benchmarks']['kernels'].items()):
            if (read_streams >= (kernel_info['read streams']['streams'] +
                                 kernel_info['write streams']['streams'] -
                                 kernel_info['read+write streams']['streams']) >
                    measurement_kernel_info['read streams']['streams'] +
                    measurement_kernel_info['write streams']['streams'] -
                    measurement_kernel_info['read+write streams']['streams'] and
                    write_streams >= kernel_info['write streams']['streams'] >
                    measurement_kernel_info['write streams']['streams']):
                measurement_kernel = kernel_name
                measurement_kernel_info = kernel_info

        # choose smt, and then use max/saturation bw
        bw_level = self['memory hierarchy'][cache_level]['level']
        bw_measurements = \
            self['benchmarks']['measurements'][bw_level][threads_per_core]
        assert threads_per_core == bw_measurements['threads per core'], \
            'malformed measurement dictionary in machine file.'
        if cores:
            # Used by Roofline model
            run_index = bw_measurements['cores'].index(cores)
            bw = bw_measurements['results'][measurement_kernel][run_index]
        else:
            # Used by ECM model
            bw = max(bw_measurements['results'][measurement_kernel])

        # Correct bandwidth due to miss-measurement of write allocation
        # TODO support non-temporal stores and non-write-allocate architectures
        measurement_kernel_info = self['benchmarks']['kernels'][measurement_kernel]
        factor = (float(measurement_kernel_info['read streams']['bytes']) +
                  2.0*float(measurement_kernel_info['write streams']['bytes']) -
                  float(measurement_kernel_info['read+write streams']['bytes'])) / \
                 (float(measurement_kernel_info['read streams']['bytes']) +
                  float(measurement_kernel_info['write streams']['bytes']))
        bw = bw * factor

        return bw, measurement_kernel
