#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division

from ruamel import yaml
import cachesim


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

    def get_cachesim(self, cores=1):
        '''Returns a cachesim.CacheSimulator object based on the machine description
        and used core count'''
        cache_stack = []
        cache = None
        cl_size = int(self['cacheline size'])

        for cache_dscr in reversed(self['memory hierarchy'][:-1]):
            # Default case, all of the cache goes to one processor
            cache_size = int(cache_dscr['size per group'])
            # reduce cache size in parallel execution
            if cores > 1 and cache_dscr['cores per group'] is not None and \
                cache_dscr['cores per group'] > 1:
                if cores < cache_dscr['cores per group']:
                    cache_size /= cores
                else:
                    cache_size /= cache_dscr['cores per group']

            if cache_dscr['ways'] is not None:
                # N-way cache associativity
                cache_ways = int(cache_dscr['ways'])
                cache_sets = cache_size // (cache_ways*cl_size)
            else:
                # Full associativity
                # this will increase the cache size to the next power of two
                cache_ways = 2**((cache_size//cl_size-1).bit_length())  # needs to be a pow. of 2
                cache_sets = 1
            cache_strategy = cache_dscr['replacement strategy']

            # print(cache_sets, cache_ways, cl_size)
            cache = cachesim.Cache(
                cache_sets, cache_ways, cl_size, cache_strategy, parent=cache)
            cache_stack.append(cache)

        mem = cachesim.MainMemory(cache_stack[0])

        mh = cachesim.CacheSimulator(cache_stack[-1], mem)

        return mh

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
