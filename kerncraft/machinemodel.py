#!/usr/bin/env python3
"""Machine model and helper functions."""
import os
from datetime import datetime
from distutils.spawn import find_executable
from distutils.version import LooseVersion
import re
from collections import OrderedDict
from copy import deepcopy
import hashlib

import ruamel
import cachesim
from sympy.parsing.sympy_parser import parse_expr

from . import prefixedunit
from . import __version__


MIN_SUPPORTED_VERSION = "0.8.1dev0"

CHANGES_SINCE = OrderedDict([
    ("0.6.6",
     """
     Removed 'cycles per cache line transfer' and replaced it by 
     'non-overlap upstream throughput' in cache levels. The new parameter
     takes the following arguments and is now associated with the cache level 
     that is read from or written to: 
     [$TP B/cy or 'full socket memory bandwidth', 'half-duplex' or 'full-duplex']
     """),
    ("0.7.1",
     """
     The dictionary under 'compiler' needs to be tagged with '!!omap' and formatted
     as a sequence. For example: '- compiler_command: arg u ment s'. Pay attention
     to the leading dash.
     """),
    ("0.8.1dev0",
     """
     Removed 'non-overlap upstream throughput' and replaced it by 
     'upstream throughput' in cache levels. This new parameter
     takes additionally the following argument: 
     ['architecture code analyzer', ['data ports' ,'list']
     New argument 'transfers overlap' in cache levels, which may be True or False.
     **Preliminary solution! Subjected to future changes.**
     """),
])


def sanitize_symbolname(name):
    """
    Sanitize all characters not matched to a symbol by sympy's parse_expr.

    Based on same rules as used for python variables.
    """
    return re.subn('(^[0-9])|[^0-9a-zA-Z_]', '_', name)[0]


class MachineModel(object):
    """Representation of the hardware and machine architecture."""

    def __init__(self, path_to_yaml=None, machine_yaml=None, args=None):
        """Create machine representation from yaml file."""
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
                self._data = ruamel.yaml.load(f, Loader=ruamel.yaml.Loader)

        assert 'kerncraft version' in self._data, \
            "Machine description requires a 'kerncraft version' entry, containg the kerncraft " \
            "version it was written for."
        file_version = LooseVersion(self._data['kerncraft version'])
        if not (MIN_SUPPORTED_VERSION <= file_version
                <= LooseVersion(__version__)):
            print("Relevant changes to the machine description file format:")
            print('\n'.join(['{}: {}'.format(version, help_text)
                             for version, help_text in CHANGES_SINCE.items()
                             if LooseVersion(version) > file_version]))
            raise ValueError("Machine description is incompatible with this version. "
                             "Supported versions are from {} to {}. Check change logs and examples "
                             "to update your own machine description file format.".format(
                                MIN_SUPPORTED_VERSION, __version__))

    def __getitem__(self, key):
        """Return configuration entry."""
        return self._data[key]

    def __contains__(self, key):
        """Return true if configuration key is present."""
        return key in self._data

    def __repr__(self):
        """Return object representation."""
        return '{}({})'.format(
            self.__class__.__name__,
            repr(self._path or self._data['model name']),
        )

    def get_identifier(self):
        """Return identifier which is either the machine file name or sha256 checksum of data."""
        if self._path:
            return os.path.basename(self._path)
        else:
            return hashlib.sha256(hashlib.sha256(repr(self._data).encode())).hexdigest()

    def get_last_modified_datetime(self):
        """Return datetime object of modified time of machine file. Return now if not a file."""
        if self._path:
            statbuf = os.stat(self._path)
            return datetime.utcfromtimestamp(statbuf.st_mtime)
        else:
            return datetime.now()

    def get_cachesim(self, cores=1):
        """
        Return a cachesim.CacheSimulator object based on the machine description.

        :param cores: core count (default: 1)
        """
        cache_dict = {}
        for c in self['memory hierarchy']:
            # Skip main memory
            if 'cache per group' not in c:
                continue
            cache_dict[c['level']] = deepcopy(c['cache per group'])
            # Scale size of shared caches according to cores
            if c['cores per group'] > 1:
                cache_dict[c['level']]['sets'] //= cores

        cs, caches, mem = cachesim.CacheSimulator.from_dict(cache_dict)

        return cs

    def get_bandwidth(self, cache_level, read_streams, write_streams, threads_per_core, cores=None):
        """
        Return best fitting bandwidth according to number of threads, read and write streams.

        :param cache_level: integer of cache (0 is L1, 1 is L2 ...)
        :param read_streams: number of read streams expected
        :param write_streams: number of write streams expected
        :param threads_per_core: number of threads that are run on each core
        :param cores: if not given, will choose maximum bandwidth for single NUMA domain
        """
        # try to find best fitting kernel (closest to read/write ratio):
        # write allocate has to be handled in kernel information (all writes are also reads)
        # TODO support for non-write-allocate architectures
        try:
            target_ratio = read_streams/write_streams
        except ZeroDivisionError:
            target_ratio = float('inf')
        measurement_kernel = 'load'
        measurement_kernel_info = self['benchmarks']['kernels'][measurement_kernel]
        measurement_kernel_ratio = float('inf')
        for kernel_name, kernel_info in sorted(self['benchmarks']['kernels'].items()):
            try:
                kernel_ratio = ((kernel_info['read streams']['streams'] +
                                 kernel_info['write streams']['streams'] -
                                 kernel_info['read+write streams']['streams']) /
                                kernel_info['write streams']['streams'])
            except ZeroDivisionError:
                kernel_ratio = float('inf')

            if abs(kernel_ratio - target_ratio) < abs(measurement_kernel_ratio - target_ratio):
                measurement_kernel = kernel_name
                measurement_kernel_info = kernel_info
                measurement_kernel_ratio = kernel_ratio

        # choose smt, and then use max/saturation bw
        bw_level = self['memory hierarchy'][cache_level]['level']
        bw_measurements = \
            self['benchmarks']['measurements'][bw_level][threads_per_core]
        assert threads_per_core == bw_measurements['threads per core'], \
            'malformed measurement dictionary in machine file.'
        if cores is not None:
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
        if cache_level == 0:
            # L1 does not have write-allocate, so everything is measured correctly
            factor = 1.0
        else:
            factor = (float(measurement_kernel_info['read streams']['bytes']) +
                      2.0*float(measurement_kernel_info['write streams']['bytes']) -
                      float(measurement_kernel_info['read+write streams']['bytes'])) / \
                     (float(measurement_kernel_info['read streams']['bytes']) +
                      float(measurement_kernel_info['write streams']['bytes']))
        bw = bw * factor

        return bw, measurement_kernel

    def get_compiler(self, compiler=None, flags=None):
        """
        Return tuple of compiler and compiler flags.

        Selects compiler and flags from machine description file, commandline arguments or call
        arguements.
        """
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
                                   "$PATH.".format(list(self['compiler'].keys())))
        if flags is None:
            # Select from machine description file
            flags = self['compiler'].get(compiler, '')

        return compiler, flags.split(' ')

    @staticmethod
    def parse_perfctr_event(perfctr):
        """
        Parse events in machine description to tuple representation used in Benchmark module.

        Examples:
        >>> parse_perfctr_event('PERF_EVENT:REG[0-3]')
        ('PERF_EVENT', 'REG[0-3]')
        >>> parse_perfctr_event('PERF_EVENT:REG[0-3]:STAY:FOO=23:BAR=0x23')
        ('PERF_EVENT', 'REG[0-3]', {'STAY': None, 'FOO': 23, 'BAR': 35})

        """
        split_perfctr = perfctr.split(':')
        assert len(split_perfctr) >= 2, "Atleast one colon (:) is required in the event name"
        event_tuple = split_perfctr[:2]
        parameters = {}
        for p in split_perfctr[2:]:
            if '=' in p:
                k, v = p.split('=')
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
        """Return (sympy expressions, event names and symbols dict) from performance metric str."""
        # Find all perfs counter references
        perfcounters = re.findall(r'[A-Z0-9_]+:[A-Z0-9\[\]|\-]+(?::[A-Za-z0-9\-_=]+)*', metric)

        # Build a temporary metric, with parser-friendly Symbol names
        temp_metric = metric
        temp_pc_names = {"SYM{}".format(re.sub("[\[\]\-|=:]", "_", pc)): pc
                         for i, pc in enumerate(perfcounters)}
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
