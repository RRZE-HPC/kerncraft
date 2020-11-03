#!/usr/bin/env python3
"""Machine model and helper functions."""
import argparse
import io
import os
import subprocess
import sys
from datetime import datetime
from distutils.spawn import find_executable
from distutils.version import LooseVersion
import re
from collections import OrderedDict
from copy import deepcopy, copy
import hashlib
from functools import lru_cache

import psutil
from ruamel import yaml
from ruamel.yaml.comments import CommentedMap
import cachesim
from sympy.parsing.sympy_parser import parse_expr

from .prefixedunit import PrefixedUnit
from . import __version__


MIN_SUPPORTED_VERSION = "0.8.3.dev0"

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
    ("0.8.1.dev0",
     """
     Removed 'non-overlap upstream throughput' and replaced it by
     'upstream throughput' in cache levels. This new parameter
     takes additionally the following argument:
     ['architecture code analyzer', 'data ports' ,'list']
     New argument 'transfers overlap' in cache levels, which may be True or False.
     **Preliminary solution! Subjected to future changes.**
     """),
    ("0.8.3.dev0",
     """
     Replaced 'micro-architecture' and 'micro-architecture modeller' with
     'in-core model' ordered map, which allows multiple model tools to be
     supported by a single machine file. The first entry is used by default.

     Also added stats to benchmark measurements for (manual) validation of model
     parameters.
     """),
    ("0.8.3.dev1",
     """
     Added ISA attribute, which may either be x86 or aarch64 (or any ISA name
     supported by OSACA)
     """),
    ("0.8.6.dev0",
    """
    Per in-core model 'port' attribute for 'overlapping model' and 
    'non-overlapping model' introduced to support LLVM-MCAs naming scheme.
    """)
])


def sanitize_symbolname(name):
    """
    Sanitize all characters not matched to a symbol by sympy's parse_expr.

    Based on same rules as used for python variables.
    """
    return re.subn('(^[0-9])|[^0-9a-zA-Z_]', '_', name)[0]


class MachineModel(object):
    """Representation of the hardware and machine architecture."""
    _loaded_machine_yaml = {}

    def __init__(self, path_to_yaml=None, machine_yaml=None, args=None):
        """
        Create machine representation from yaml file or current system

        :param path_to_yaml: path to YAML machine file
        :param machine_yaml: string containing YAML machine information

        One or the other needs to be passed. If none is given

        """
        self._data = dict([
            ('kerncraft version', __version__),
            ('model type', 'INFORMATION_REQUIRED'),
            ('model name', 'INFORMATION_REQUIRED'),
            ('sockets', 'INFORMATION_REQUIRED'),
            ('cores per socket', 'INFORMATION_REQUIRED'),
            ('threads per core', 'INFORMATION_REQUIRED'),
            ('NUMA domains per socket', 'INFORMATION_REQUIRED'),
            ('cores per NUMA domain', 'INFORMATION_REQUIRED'),
            ('clock', 'INFORMATION_REQUIRED (e.g., 2.7 GHz)'),
            ('FLOPs per cycle', {'SP': {'total': 'INFORMATION_REQUIRED',
                                        'FMA': 'INFORMATION_REQUIRED',
                                        'ADD': 'INFORMATION_REQUIRED',
                                        'MUL': 'INFORMATION_REQUIRED'},
                                 'DP': {'total': 'INFORMATION_REQUIRED',
                                        'FMA': 'INFORMATION_REQUIRED',
                                        'ADD': 'INFORMATION_REQUIRED',
                                        'MUL': 'INFORMATION_REQUIRED'}}),
            ('isa', 'INFORMATION_REQUIRED (e.g., x86, aarch64)'),
            ('in-core model', OrderedDict([
                ('IACA', 'INFORMATION_REQUIRED (e.g., NHM, WSM, SNB, IVB, HSW, BDW, SKL, SKX)'),
                ('OSACA', 'INFORMATION_REQUIRED (e.g., NHM, WSM, SNB, IVB, HSW, BDW, SKL, SKX)'),
                ('LLVM-MCA', 'INFORMATION_REQUIRED (e.g., -mcpu=skylake-avx512)')])),
            ('compiler', OrderedDict([
                ('icc', 'INFORMATION_REQUIRED (e.g., -O3 -fno-alias -xAVX)'),
                ('clang', 'INFORMATION_REQUIRED (e.g., -O3 -mavx, -D_POSIX_C_SOURCE=200112L, check '
                          '`gcc -march=native -Q --help=target | grep -- "-march="`)'),
                ('gcc', 'INFORMATION_REQUIRED (e.g., -O3 -march=ivybridge, check `gcc -march=native -Q '
                        '--help=target | grep -- "-march="`)')])),
            ('cacheline size', 'INFORMATION_REQUIRED (in bytes, e.g. 64 B)'),
            ('overlapping model', {
                'ports': 'INFORMATION_REQUIRED (list of ports as they appear in IACA, e.g.,'
                         ', ["0", "0DV", "1", "2", "2D", "3", "3D", "4", "5", "6", "7"])',
                'performance counter metric':
                    'INFORMATION_REQUIRED Example:'
                    'max(UOPS_DISPATCHED_PORT_PORT_0__PMC2, UOPS_DISPATCHED_PORT_PORT_1__PMC3,'
                    '    UOPS_DISPATCHED_PORT_PORT_4__PMC0, UOPS_DISPATCHED_PORT_PORT_5__PMC1)'
            }),
            ('non-overlapping model', {
                'ports': 'INFORMATION_REQUIRED (list of ports as they appear in IACA, e.g.,'
                         ', ["0", "0DV", "1", "2", "2D", "3", "3D", "4", "5", "6", "7"])',
                'performance counter metric':
                    'INFORMATION_REQUIRED Example:'
                    'max(UOPS_DISPATCHED_PORT_PORT_0__PMC2, UOPS_DISPATCHED_PORT_PORT_1__PMC3,'
                    '    UOPS_DISPATCHED_PORT_PORT_4__PMC0, UOPS_DISPATCHED_PORT_PORT_5__PMC1)'
            }),
            ('memory hierarchy', 'INFORMATION_REQUIRED'),
            ('benchmarks', 'INFORMATION_REQUIRED'),
            ('machine state', 'INFORMATION_REQUIRED (output of machine-state.sh)'),
        ])

        if path_to_yaml and machine_yaml:
            raise ValueError('Only one of path_to_yaml and machine_yaml is allowed')
        self._path = path_to_yaml
        self._args = args
        if path_to_yaml:
            # Load into cache and save to self._data
            abspath_to_yaml = os.path.abspath(path_to_yaml)
            if abspath_to_yaml not in self._loaded_machine_yaml:
                with open(path_to_yaml, 'r') as f:
                    # Ignore ruamel unsafe loading warning, by supplying Loader parameter
                    self._loaded_machine_yaml[abspath_to_yaml] = yaml.load(f, Loader=yaml.Loader)
            self._data = self._loaded_machine_yaml[abspath_to_yaml]
        elif machine_yaml:
            self._data = machine_yaml

        assert 'kerncraft version' in self._data, \
            "Machine description requires a 'kerncraft version' entry, containg the kerncraft " \
            "version it was written for."
        file_version = LooseVersion(self._data['kerncraft version'])
        if not (LooseVersion(MIN_SUPPORTED_VERSION) <= file_version
                <= LooseVersion(__version__)):
            print("Relevant changes to the machine description file format:")
            print('\n'.join(['{}: {}'.format(version, help_text)
                             for version, help_text in CHANGES_SINCE.items()
                             if LooseVersion(version) > file_version]))
            raise ValueError("Machine description is incompatible with this version. "
                             "Supported versions are from {} to {}. Check change logs and examples "
                             "to update your own machine description file format.".format(
                                MIN_SUPPORTED_VERSION, __version__))

    def set_path(self, path):
        self._path = path

    def update(self, readouts=True, memory_hierarchy=True, benchmarks=True, overwrite=True,
               machine_state=True, cpuinfo_path: str='/proc/cpuinfo'):
        """Update model from readouts and benchmarks on current machine."""
        if readouts:
            self._data.update(get_machine_readouts(cpuinfo_path=cpuinfo_path))
        if memory_hierarchy:
            self._data.update(get_memory_hierarchy(placeholders=overwrite,
                                                   cpuinfo_path=cpuinfo_path))

        if benchmarks:
            self._update_benchmarks()
        
        if machine_state:
            self._update_machine_state()

    def _update_benchmarks(self, repetitions=10,
                           usage_factor=0.66, min_surpass_factor=0.2, mem_factor=15.0,
                           overwrite=False):
        """Run benchmarks and update internal dataset"""
        if not isinstance(self._data['benchmarks'], dict):
            self._data['benchmarks'] = {}
        benchmarks = self._data['benchmarks']

        benchmark_kernels = {
                'load': {
                    'read streams': {'streams': 1, 'bytes': PrefixedUnit(8, 'B')},
                    'read+write streams': {'streams': 0, 'bytes': PrefixedUnit(0, 'B')},
                    'write streams': {'streams': 0, 'bytes': PrefixedUnit(0, 'B')},
                    'FLOPs per iteration': 0},
                'copy': {
                    'read streams': {'streams': 1, 'bytes': PrefixedUnit(8, 'B')},
                    'read+write streams': {'streams': 0, 'bytes': PrefixedUnit(0, 'B')},
                    'write streams': {'streams': 1, 'bytes': PrefixedUnit(8, 'B')},
                    'FLOPs per iteration': 0},
                'update': {
                    'read streams': {'streams': 1, 'bytes': PrefixedUnit(8, 'B')},
                    'read+write streams': {'streams': 1, 'bytes': PrefixedUnit(8, 'B')},
                    'write streams': {'streams': 1, 'bytes': PrefixedUnit(8, 'B')},
                    'FLOPs per iteration': 0},
                'triad': {
                    'read streams': {'streams': 3, 'bytes': PrefixedUnit(24, 'B')},
                    'read+write streams': {'streams': 0, 'bytes': PrefixedUnit(0, 'B')},
                    'write streams': {'streams': 1, 'bytes': PrefixedUnit(8, 'B')},
                    'FLOPs per iteration': 2},
                'daxpy': {
                    'read streams': {'streams': 2, 'bytes': PrefixedUnit(16, 'B')},
                    'read+write streams': {'streams': 1, 'bytes': PrefixedUnit(8, 'B')},
                    'write streams': {'streams': 1, 'bytes': PrefixedUnit(8, 'B')},
                    'FLOPs per iteration': 2}, }

        if 'kernels' not in benchmarks:
            benchmarks['kernels'] = benchmark_kernels
        else:
            for kernel in benchmark_kernels:
                if kernel not in benchmarks['kernels']:
                    benchmarks['kernels'][kernel] = benchmark_kernels[kernel]

        if 'measurements' not in benchmarks:
            benchmarks['measurements'] = {}

        cores = list(range(1, self['cores per socket'] + 1))
        for mem_index, mem in enumerate(self['memory hierarchy']):
            try:
                measurement = benchmarks['measurements'][mem['level']]
            except (KeyError, TypeError):
                measurement = benchmarks['measurements'][mem['level']] = {}

            if mem_index > 0:
                mem_previous = self['memory hierarchy'][mem_index - 1]
            else:
                mem_previous = {
                    'size per group': 0,
                    'cores per group': 1
                }

            for threads_per_core in range(1, self['threads per core'] + 1):
                threads = [c * threads_per_core for c in cores]
                if mem['size per group'] is not None:
                    total_sizes = [
                        PrefixedUnit(max(
                            max(int(mem['size per group']) * c / mem['cores per group'],
                                int(mem['size per group'])) * usage_factor,
                            max(int(mem_previous['size per group']) * c / mem_previous['cores per group'],
                                int(mem_previous['size per group'])) * (1.0 + min_surpass_factor)), 'B')
                        for c in cores]
                else:
                    last_mem = self['memory hierarchy'][-2]
                    total_sizes = [last_mem['size per group'] * mem_factor for c in cores]
                sizes_per_core = [t / cores[i] for i, t in enumerate(total_sizes)]
                sizes_per_thread = [t / threads[i] for i, t in enumerate(total_sizes)]

                sizes_dict = {
                    'threads per core': threads_per_core,
                    'cores': copy(cores),
                    'threads': threads,
                    'size per core': sizes_per_core,
                    'size per thread': sizes_per_thread,
                    'total size': total_sizes}

                needs_update = False
                if threads_per_core in measurement:
                    for k, v in sizes_dict.items():
                        if k in measurement[threads_per_core]:
                            if v == sizes_dict[k]:
                                # Exact compartison matched
                                continue
                            for i, j in zip(v, sizes_dict[k]):
                                # Fuzzy comparison with relative error tolerance of 1%
                                if abs(i - j)/min(i, j) >= 0.01:
                                    needs_update = True
                                    break
                        else:
                            # If k is missing in measurement, will need to overwrite
                            needs_update = True
                            break

                if overwrite or threads_per_core not in measurement or needs_update or \
                        'results' not in measurement[threads_per_core] or \
                        'stats' not in measurement[threads_per_core]:
                    measurement[threads_per_core] = sizes_dict
                    # Invalidate results and stats
                    measurement[threads_per_core]['results'] = {}
                    measurement[threads_per_core]['stats'] = {}
                else:
                    # No need to change anything
                    pass

        if self._args:
            verbose = self._args.verbose
        else:
            verbose = 0

        if verbose:
            print('Progress: ', file=sys.stderr)
            sys.stderr.flush()

        for kernel in sorted(list(benchmarks['kernels'].keys())):
            # Select fastest kernel version
            if 'fastest bench kernel' not in benchmarks['kernels'][kernel] or \
                    benchmarks['kernels'][kernel]['fastest bench kernel'] is None:
                mem_level = 'L1'
                fastest_kernel = find_fastest_bench_kernel(
                    get_available_bench_kernels(prefix=kernel, excludes=['_mem', '_sp', '_nt']),
                    total_size=int(float(
                        benchmarks['measurements'][mem_level][1]['total size'][0]) / 1000),
                    threads_per_core=1,
                    max_threads_per_core=self['threads per core'],
                    cores_per_socket=1,
                    sockets=1,
                    verbose=verbose > 1)

                benchmarks['kernels'][kernel]['fastest bench kernel'] = fastest_kernel
            else:
                fastest_kernel = benchmarks['kernels'][kernel]['fastest bench kernel']

            if verbose > 1:
                print('Selected {} as fastest bench kernel for {}'.format(fastest_kernel, kernel),
                      file=sys.stderr)

            # Run actual benchmarks and safe machine file in between
            for mem_level in sorted(list(benchmarks['measurements'].keys())):
                if verbose > 1:
                    print('Running for {}'.format(mem_level), file=sys.stderr)
                for threads_per_core in sorted(list(benchmarks['measurements'][mem_level].keys())):
                    measurement = benchmarks['measurements'][mem_level][threads_per_core]
                    if overwrite or kernel not in measurement['results'] or \
                            kernel not in measurement['stats'] or \
                            not (len(measurement['results'][kernel]) ==
                                 len(measurement['stats'][kernel]) ==
                                 len(measurement['total size'])):
                        measurement['results'][kernel] = []
                        measurement['stats'][kernel] = []

                    for i, total_size in enumerate(measurement['total size']):
                        if len(measurement['results'][kernel]) > i:
                            # Skip already existing data
                            continue
                        stats = []
                        for r in range(repetitions):
                            stats.append(measure_bw(
                                         fastest_kernel,
                                         int(float(total_size) / 1000),
                                         threads_per_core,
                                         self['threads per core'],
                                         measurement['cores'][i],
                                         sockets=1,
                                         verbose=verbose > 1))

                        measurement['results'][kernel].append(copy(max(stats)))
                        measurement['stats'][kernel].append(stats)

                        self.dump()

                        if not verbose:
                            print('.', end='', file=sys.stderr)
                        sys.stderr.flush()
    
    def _update_machine_state(self):
        """Read and update machine state."""
        self._data['machine state'] = get_machien_state()

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
            cache_dict[c['level']]['cl_size'] = int(c['cache per group']['cl_size'])
            # Scale size of last cache according to cores (typically shared within NUMA domain)
            if c['cores per group'] > 1:
                cache_dict[c['level']]['sets'] //= min(cores, self['cores per NUMA domain'])

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
    
    def get_incore_model(self, model=None):
        """
        Return incore model name to use.
        """
        if model is None:
            model = next(iter(self['in-core model']))
        return model

    def current_system(self, print_diff=False):
        """
        Check if current system is same as machine model (specs and configuration)

        Does not check frequency! This needs to be done during runtime with likwid-perfctr.

        :param print_diff: print which data differs if True

        :return: True if it is the same
        """
        current_topology = get_machine_readouts()
        current_topology.update(get_memory_hierarchy())
        same = True
        for k in ['model type', 'model name', 'sockets', 'cores per socket', 'threads per core',
                  'NUMA domains per socket', 'cores per NUMA domain', 'transparent hugepage']:
            if current_topology[k] != self[k]:
                if print_diff:
                    print("Expected {!r} and found {!r} for key {!r}.".format(
                        self[k], current_topology[k], k))
                same = False
        return same

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

    def dump(self, f=None):
        """
        Return YAML string to store machine model and store to f (if path or fp passed).
        """
        yaml_string = yaml.dump(self._data, Dumper=yaml.Dumper)
        if f is None:
            f = self._path

        if isinstance(f, io.IOBase):
            f.write(yaml_string)
        else:
            with open(f, 'w') as fp:
                fp.write(yaml_string)

        return yaml_string


def get_match_or_break(regex, haystack, flags=re.MULTILINE):
    m = re.search(regex, haystack, flags)
    if not m:
        raise ValueError("could not find " + repr(regex) + " in " + repr(haystack))
    return m.groups()


def get_match_or_default(regex, haystack, default=None, flags=re.MULTILINE):
    try:
        return get_match_or_break(regex, haystack, flags)
    except ValueError:
        return default


@lru_cache(1)
def get_likwid_topology() -> str:
    topo = subprocess.check_output(['likwid-topology']).decode("utf-8")
    return topo


@lru_cache(1)
def read_cpuinfo(cpuinfo_path: str='/proc/cpuinfo') -> str:
    with open(cpuinfo_path, 'r') as f:
        cpuinfo = f.read()
    return cpuinfo


def get_cpu_frequency():
    """Get CPU frequency in Hz"""
    # TODO use likwid to read actual base frequency
    cpu_frequency = psutil.cpu_freq()
    if cpu_frequency:
        return cpu_frequency.current*1e6
    else:
        return None


def get_machien_state():
    """
    Build complete machine state information
    
    Using:
    https://github.com/RRZE-HPC/Artifact-description/blob/master/machine-state.sh
    """
    return subprocess.check_output(
        os.path.join(os.path.dirname(__file__), 'scripts', 'machine-state.sh'),
        stderr=subprocess.STDOUT).decode("utf-8")


@lru_cache(1)
def get_machine_readouts(cpuinfo_path: str='/proc/cpuinfo'):
    """Read machine information using different commands and files and return dictionary."""
    topology = get_likwid_topology()
    cpu_info = read_cpuinfo(cpuinfo_path=cpuinfo_path)

    readouts = {'kerncraft version': __version__,
                'model type': get_match_or_break(r'^CPU type:\s+(.+?)\s*$', topology)[0],
                'model name': get_match_or_default(r'^model name\s+:\s+(.+?)\s*$', cpu_info,
                                                   default=[None])[0],
                'threads per core': int(
                    get_match_or_break(r'^Threads per core:\s+([0-9]+)\s*$', topology)[0]),
                'sockets': int(get_match_or_break(r'^Sockets:\s+([0-9]+)\s*$', topology)[0]),
                'cores per socket': int(
                    get_match_or_break(r'^Cores per socket:\s+([0-9]+)\s*$', topology)[0]),}
    readouts['NUMA domains per socket'] = int(
        get_match_or_break(r'^NUMA domains:\s+([0-9]+)\s*$', topology)[0]) // readouts['sockets']
    if readouts['NUMA domains per socket'] == 0:
        readouts['NUMA domains per socket'] = 1
    readouts['cores per NUMA domain'] = \
        readouts['cores per socket'] // readouts['NUMA domains per socket']
    clock = get_cpu_frequency()
    if clock is not None:
        readouts['clock'] = PrefixedUnit(clock, "Hz")
    with open('/sys/kernel/mm/transparent_hugepage/enabled') as f:
        readouts['transparent hugepage'] = get_match_or_break(r'\[([a-z]+)\]', f.read())[0]

    return readouts


@lru_cache(1)
def get_memory_hierarchy(placeholders=True, cpuinfo_path: str='/proc/cpuinfo'):
    """Read cache hierarchy using different commands and files and return dictionary."""
    readouts = get_machine_readouts(cpuinfo_path=cpuinfo_path)
    topology = get_likwid_topology()

    threads_start = topology.find('HWThread')
    threads_end = topology.find('Cache Topology')
    threads = {}
    for line in topology[threads_start:threads_end].split('\n'):
        m = re.match(r'([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)', line)
        if m:
            threads[m.groups()[0]] = (m.groups()[1:])

    cache_start = topology.find('Cache Topology')
    cache_end = topology.find('NUMA Topology')
    memory_hierarchy = []
    mem_level = OrderedDict()
    for line in topology[cache_start:cache_end].split('\n'):
        if line.startswith('Level:'):
            mem_level = OrderedDict([('level', 'L' + line.split(':')[1].strip())])
            memory_hierarchy.append(mem_level)
            if mem_level['level'] != 'L1' and placeholders:
                mem_level['upstream throughput'] = [
                    'INFORMATION_REQUIRED (e.g. 24 B/cy)',
                    'INFORMATION_REQUIRED (e.g. "half-duplex" or "full-duplex")']
        elif line.startswith('Size:'):
            size = PrefixedUnit(line.split(':')[1].strip())
            if placeholders:
                mem_level['cache per group'] = OrderedDict([
                    ('sets', 'INFORMATION_REQUIRED (sets*ways*cl_size=' + str(size) + ')'),
                    ('ways', 'INFORMATION_REQUIRED (sets*ways*cl_size=' + str(size) + ')'),
                    ('cl_size', 'INFORMATION_REQUIRED (sets*ways*cl_size=' + str(size) + ')'),
                    ('replacement_policy', 'INFORMATION_REQUIRED (options: LRU, FIFO, MRU, RR)'),
                    ('write_allocate', 'INFORMATION_REQUIRED (True/False)'),
                    ('write_back', 'INFORMATION_REQUIRED (True/False)'),
                ])
            mem_level['cache per group']['load_from'] = 'L' + str(int(mem_level['level'][1:]) + 1)
            mem_level['cache per group']['store_to'] = 'L' + str(int(mem_level['level'][1:]) + 1)
            mem_level['size per group'] = size
        elif line.startswith('Cache groups:'):
            mem_level['groups'] = line.count('(')
            mem_level['cores per group'] = \
                (readouts['cores per socket'] * readouts['sockets']) // mem_level['groups']
            mem_level['threads per group'] = \
                int(mem_level['cores per group'] * readouts['threads per core'])
        if placeholders:
            mem_level['performance counter metrics'] = {
                'accesses': 'INFORMATION_REQUIRED (e.g., L1D_REPLACEMENT__PMC0)',
                'misses': 'INFORMATION_REQUIRED (e.g., L2_LINES_IN_ALL__PMC1)',
                'evicts': 'INFORMATION_REQUIRED (e.g., L2_LINES_OUT_DIRTY_ALL__PMC2)'
            }

    # Remove last caches load_from and store_to:
    del memory_hierarchy[-1]['cache per group']['load_from']
    del memory_hierarchy[-1]['cache per group']['store_to']

    memory_hierarchy.append(OrderedDict([
        ('level', 'MEM'),
        ('cores per group', int(readouts['cores per socket'])),
        ('threads per group', int(readouts['threads per core'] * readouts['cores per socket'])),
    ]))
    if placeholders:
        memory_hierarchy[-1]['upstream throughput'] = [
            'full socket memory bandwidth',
            'INFORMATION_REQUIRED (e.g. "half-duplex" or "full-duplex")']
    memory_hierarchy[-1]['penalty cycles per cacheline load'] = 0
    memory_hierarchy[-1]['penalty cycles per cacheline store'] = 0
    memory_hierarchy[-1]['size per group'] = None

    return {'memory hierarchy': memory_hierarchy}


def measure_bw(kernel, total_size, threads_per_core, max_threads_per_core, cores_per_socket,
               sockets, verbose=False):
    """*size* is given in kilo bytes"""

    groups = []
    for s in range(sockets):
        groups += [
            '-W',
            'S' + str(s) + ':' + str(total_size) + 'kB:' +
            str(threads_per_core * cores_per_socket) +
            ':1:' + str(int(max_threads_per_core / threads_per_core))]
    # for older likwid versions add ['-g', str(sockets), '-i', str(iterations)] to cmd
    cmd = ['likwid-bench', '-t', kernel] + groups
    if verbose:
        print('{:<50} = '.format(' '.join(cmd)), end='', file=sys.stderr)

    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode('utf-8')
    if not output:
        print(' '.join(cmd) + ' returned no output, possibly wrong version installed '
                              '(requires 5.0 or later)', file=sys.stderr)
        sys.exit(1)
    bw = float(get_match_or_break(r'^MByte/s:\s+([0-9]+(?:\.[0-9]+)?)\s*$', output)[0])
    if verbose:
        print(PrefixedUnit(bw, 'MB/s'), file=sys.stderr)

    return PrefixedUnit(bw, 'MB/s')


def find_fastest_bench_kernel(kernels, *args, **kwargs):
    """
    Measure and return fastest kernel

    :param kernels: list of kernels to test
    all other arguments will be passed onto `measure_bw(...)`
    """
    results = []
    for k in kernels:
        try:
            results.append((measure_bw(k, *args, **kwargs), k))
        except (ValueError, subprocess.CalledProcessError):
            # Ignore failed likwid-bench runs, because some kernels may not be supported on
            # all architectures (e.g., avx512 is not supported on Sandy Bridge)
            if 'verbose' in kwargs and kwargs['verbose']:
                print(file=sys.stderr)
            pass

    return max(results)[1]


def get_available_bench_kernels(prefix="", excludes=[]):
    """
    Return list of available likwid-bench kernels
    :param prefix: only return kernels which start with this prefix
    :param exclude: list of substrings, which must not be found in kernel name
    :return: list of strings
    """
    output = subprocess.check_output(['likwid-bench', '-a']).decode('utf-8').strip()
    output = [l.split(' - ')[0] for l in output.split('\n')]

    result = []
    for l in output:
        # Check if prefix matches
        if l.startswith(prefix):
            # Check each exclude
            skip = False
            for e in excludes:
                if e in l:
                    skip = True
                    break
            if not skip:
                result.append(l)
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Machine description file generator.',
        epilog='For help, examples, documentation and bug reports go to:\nhttps://github.com'
               '/RRZE-HPC/kerncraft\nLicense: AGPLv3', )
    # parser.add_argument('--version', action=VersionAction, version='{}'.format(__version__))
    parser.add_argument('--machine', '-m', type=argparse.FileType('r'), required=False,
                        help='Path to machine description yaml file as basis for new file.')
    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help='Increases verbosity level.')
    parser.add_argument('--readouts', dest='readouts', action='store_true')
    parser.add_argument('--no-readouts', dest='readouts', action='store_false')
    parser.add_argument('--memory-hierarchy', dest='memory_hierarchy', action='store_true')
    parser.add_argument('--no-memory-hierarchy', dest='memory_hierarchy', action='store_false')
    parser.add_argument('--benchmarks', dest='benchmarks', action='store_true')
    parser.add_argument('--no-benchmarks', dest='benchmarks', action='store_false')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true')
    parser.add_argument('--no-overwrite', dest='overwrite', action='store_false')
    parser.add_argument('--compare-host', action='store_true',
                        help='Compares machine file (require --machine) with current hosts and '
                              'reports if system differs in a configuration.')
    parser.add_argument('output_file', metavar='FILE', type=argparse.FileType('w'), default='-',
                        help='File to save new machine description to.')

    parser.set_defaults(readouts=True, memory_hierarchy=True, benchmarks=True, overwrite=True)

    args = parser.parse_args()

    if args.compare_host:
        if not args.machine:
            raise argparse.ArgumentError("--compare-host requires --machine")
        m = MachineModel(args.machine.name)
        if m.current_system(print_diff=True):
            sys.exit(0)
        else:
            sys.exit(1)

    if args.machine:
        m = MachineModel(args.machine.name, args=args)
    else:
        m = MachineModel(args=args)

    m.set_path(args.output_file.name)

    try:
        m.update(readouts=args.readouts, memory_hierarchy=args.memory_hierarchy,
                 benchmarks=args.benchmarks, overwrite=args.overwrite)
    except KeyboardInterrupt:
        print("Incomplete machine file was written. Continue by providing it via -m argument.")
    
    m.dump()


if __name__ == '__main__':
    main()
