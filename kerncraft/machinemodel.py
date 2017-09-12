#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division

from distutils.spawn import find_executable
import re

import ruamel
import cachesim
from sympy.parsing.sympy_parser import parse_expr

from . import prefixedunit


def sanitize_symbolname(name):
    '''
    Sanitizes all characters not matched to a symbol by sympy's parse_expr
    (same rules apply as for python variables)
    '''
    return re.subn('(^[0-9])|[^0-9a-zA-Z_]', '_', name)[0]

class MachineModel(object):
    def __init__(self, path_to_yaml=None, machine_yaml=None, args=None):
        if not path_to_yaml and not machine_yaml:
            raise ValueError('Either path_to_yaml or machine_yaml is required')
        if path_to_yaml and machine_yaml:
            raise ValueError('Only one of path_to_yaml and machine_yaml is allowed')
        self._path = path_to_yaml
        self._data = machine_yaml
        self._args = args
        if path_to_yaml:
            with open(path_to_yaml, 'r') as f:
                # Ignore ruamel unsafe loading warning, by supplying Loader parameter
                self._data = ruamel.yaml.load(f, Loader=ruamel.yaml.RoundTripLoader)

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
            # Choose maximum number of cores to get bandwidth for
            max_cores = min(self['memory hierarchy'][cache_level]['cores per group'],
                            self['cores per NUMA domain'])
            bw = max(bw_measurements['results'][measurement_kernel][:max_cores])

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

    def get_compiler(self, compiler=None, flags=None):
        '''
        Returns tuple of compiler and compiler flags

        Selects compiler and flags from machine description file, commandline arguments or params
        '''
        if self._args:
            compiler = compiler or self._args.compiler
            flags = flags or self._args.compiler_flags
        if compiler is None:
            # Select first available compiler in machine description file's compiler dict
            for c in self['compiler'].keys():
                # Making sure compiler is available:
                if find_executable(c) is not None:
                    compiler = c
                    break
            else:
                raise RuntimeError("No compiler ({}) was found. Add different one in machine file, "
                                   "via --compiler argument or make sure it will be found in "
                                   "$PATH.".format(list(self['compiler'].keys())), file=sys.stderr)
        if flags is None:
            # Select from machine description file
            flags = self['compiler'].get(compiler, '')

        return compiler, flags.split(' ')

    @staticmethod
    def parse_perfctr_event(perfctr):
        '''
        Parses events in machine description to tuple representation used in Benchmark module

        Examples:
        >>> parse_perfctr_event('PERF_EVENT:REG[0-3]')
        ('PERF_EVENT', 'REG[0-3]', None)
        >>> parse_perfctr_event('PERF_EVENT:REG[0-3]:STAY:FOO=23:BAR=0x23')
        ('PERF_EVENT', 'REG[0-3]', {'STAY': None, 'FOO': 23, 'BAR': 35})
        '''
        split_perfctr = perfctr.split(':')
        assert len(split_perfctr) >= 2, "Atleast one colon (:) is required in the event name"
        event_tuple = split_perfctr[:2]
        parameters = {}
        for p in split_perfctr[2:]:
            if '=' in p:
                k,v = p.split('=')
                if v.startswith('0x'):
                    parameters[k] = int(v, 16)
                else:
                    parameters[k] = int(v)
            else:
                parameters[p] = None
        event_tuple.append(parameters)
        return tuple(event_tuple)

    @staticmethod
    def parse_perfmetric(metric):
        '''
        Takes a performance metric describing string and constructs sympy and perf. counter
        representation from it.

        Returns a tuple containing the sympy expression and a dict with performance counters and
        symbols the expression depends on.
        '''
        # Find all perfs counter references
        perfcounters = re.findall(r'[A-Z0-9_]+:[A-Z0-9\[\]\|\-]+(?::[A-Za-z0-9\-_=]+)*', metric)

        # Build a temporary metric, with parser-friendly Symbol names
        temp_metric = metric
        temp_pc_names = {sanitize_symbolname(pc): pc for pc in perfcounters}
        for var_name, pc in temp_pc_names.items():
            temp_metric = temp_metric.replace(pc, var_name)
        # Parse temporary expression
        expr = parse_expr(temp_metric)

        # Rename symbols to originals
        for s in expr.free_symbols:
            if s.name in temp_pc_names:
                s.name = temp_pc_names[str(s)]

        events = {s: MachineModel.parse_perfctr_event(s.name) for s in expr.free_symbols
                  if s.name in perfcounters}

        return expr, events

