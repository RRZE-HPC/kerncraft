#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import

# Version check
import sys
if sys.version_info[0] == 2 and sys.version_info < (2, 6) or \
        sys.version_info[0] == 3 and sys.version_info < (3, 4):
    print("Must use python 2.6 or 3.4 and greater.", file=sys.stderr)
    sys.exit(1)

import subprocess
import re
from copy import copy
from pprint import pprint

from ruamel import yaml

from .prefixedunit import PrefixedUnit
from six.moves import range


def get_match_or_break(regex, haystack, flags=re.MULTILINE):
    m = re.search(regex, haystack, flags)
    if not m:
        raise ValueError("could not find "+repr(regex)+" in "+repr(haystack))
    return m.groups()


def get_machine_topology():
    try:
        topo = subprocess.Popen(['likwid-topology'], stdout=subprocess.PIPE).communicate()[0]
    except OSError as e:
        print('likwid-topology execution failed, is it installed and loaded?', file=sys.stderr)
        sys.exit(1)
    cpuinfo = open('/proc/cpuinfo', 'r').read()
    machine = {
        'model type': get_match_or_break(r'^CPU type:\s+(.+?)\s*$', topo)[0],
        'model name': get_match_or_break(r'^model name\s+:\s+(.+?)\s*$', cpuinfo)[0],
        'sockets': int(get_match_or_break(r'^Sockets:\s+([0-9]+)\s*$', topo)[0]),
        'cores per socket': int(get_match_or_break(r'^Cores per socket:\s+([0-9]+)\s*$', topo)[0]),
        'threads per core': int(get_match_or_break(r'^Threads per core:\s+([0-9]+)\s*$', topo)[0]),
        'clock': 'INFORMATION_REQUIRED (e.g., 2.7 GHz)',
        'FLOPs per cycle': {'SP': {'total': 'INFORMATION_REQUIRED',
                                   'FMA': 'INFORMATION_REQUIRED',
                                   'ADD': 'INFORMATION_REQUIRED',
                                   'MUL': 'INFORMATION_REQUIRED'},
                            'DP': {'total': 'INFORMATION_REQUIRED',
                                   'FMA': 'INFORMATION_REQUIRED',
                                   'ADD': 'INFORMATION_REQUIRED',
                                   'MUL': 'INFORMATION_REQUIRED'}},
        'micro-architecture': 'INFORMATION_REQUIRED (options: NHM, WSM, SNB, IVB, HSW)',
        'compiler': 'INFORMATION_REQUIRED (e.g., gcc)',
        'compiler flags': 'INFORMATION_REQUIRED (list of flags, e.g., [-O3, -xACX, -fno-alias])',
        'cacheline size': 'INFORMATION_REQUIRED (in bytes, e.g. 64 B)',
        'overlapping ports': 'INFORAMTION_REQUIRED (list of ports as they appear in IACA, e.g.)' + \
                             ', ["0", "0DV", "1", "2", "3", "4", "5", "6", "7"])',
        'non-overlapping ports': 'INFORMATION_REQUIRED (like overlapping ports)',
    }

    threads_start = topo.find('HWThread')
    threads_end = topo.find('Cache Topology')
    threads = {}
    for line in topo[threads_start:threads_end].split('\n'):
        m = re.match(r'([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)', line)
        if m:
            threads[m.groups()[0]] = (m.groups()[1:])

    cache_start = topo.find('Cache Topology')
    cache_end = topo.find('NUMA Topology')
    machine['memory hierarchy'] = []
    mem_level = {}
    for line in topo[cache_start:cache_end].split('\n'):
        if line.startswith('Level:'):
            mem_level = {}
            mem_level['level'] = 'L'+line.split(':')[1].strip()
            machine['memory hierarchy'].append(mem_level)
        elif line.startswith('Size:'):
            size = PrefixedUnit(line.split(':')[1].strip())
            mem_level['cache per group'] = {
                'sets': 'INFORMATION_REQUIRED (sets*ways*cl_size='+str(size)+')',
                'ways': 'INFORMATION_REQUIRED (sets*ways*cl_size='+str(size)+')',
                'cl_size': 'INFORMATION_REQUIRED (sets*ways*cl_size='+str(size)+')',
                'replacement_policy': 'INFORMATION_REQUIRED (options: LRU, FIFO, MRU, RR)',
                'write_allocate': 'INFORMATION_REQUIRED (True/False)',
                'write_back': 'INFORMATION_REQUIRED (True/False)',
                'load_from': 'L'+str(int(mem_level['level'][1:])+1),
                'store_to': 'L'+str(int(mem_level['level'][1:])+1)}
            mem_level['size per group'] = size
        elif line.startswith('Cache groups:'):
            mem_level['groups'] = line.count('(')
            mem_level['cores per group'] = \
                (machine['cores per socket'] * machine['sockets']) / mem_level['groups']
            mem_level['threads per group'] = \
                mem_level['cores per group'] * machine['threads per core']
        mem_level['cycles per cacheline transfer'] = 'INFORMATION_REQUIRED'

    # Remove last caches load_from and store_to:
    del machine['memory hierarchy'][-1]['cache per group']['load_from']
    del machine['memory hierarchy'][-1]['cache per group']['store_to']
    
    machine['memory hierarchy'].append({
        'level': 'MEM',
        'cores per group': machine['cores per socket'],
        'threads per group': machine['threads per core'] * machine['cores per socket'],
        'cycles per cacheline transfer': None,
        'penalty cycles per read stream': 0,
        'size per group': None
    })

    return machine


def measure_bw(type_, total_size, threads_per_core, max_threads_per_core, cores_per_socket,
               sockets):
    """*size* is given in kilo bytes"""
    groups = []
    for s in range(sockets):
        groups += [
            '-w',
            'S' + str(s) + ':' + str(total_size) + 'kB:' +
            str(threads_per_core * cores_per_socket) +
            ':1:'+str(int(max_threads_per_core/threads_per_core))]
    # for older likwid versions add ['-g', str(sockets), '-i', str(iterations)] to cmd
    cmd = ['likwid-bench', '-t', type_]+groups
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()[0]
    if not output:
        print(' '.join(cmd) + ' returned no output, possibly wrong version installed '
              '(requires 4.0 or later)', file=sys.stderr)
        sys.exit(1)
    bw = float(get_match_or_break(r'^MByte/s:\s+([0-9]+(?:\.[0-9]+)?)\s*$', output)[0])
    return PrefixedUnit(bw, 'MB/s')


def cli():
    # TODO support everything described here
    if '-h' in sys.argv[1:] or '--help' in sys.argv[1:]:
        print('''Usage:', sys.argv[0], '[-h] {collect|measure} [machinefile] | upgrade machinefile

        collect will retriev as much hardware information as possible, without benchmarking
        measure will do the same as collect, but also include memory benchmarks

        If machinefile already exists the CPU name will be compared. If they matche, measurements
        will proceed and the file is updated accordingly. All other information in the file
        (typically manually inserted) will be left alone.

        If no machinefile is given, the information will be printed to stdout.

        updgrade will transform machinefile to the most up-to-date machine file version.
        ''')


def main():
    machine = get_machine_topology()
    pprint(machine)

    benchmarks = {'kernels': {}, 'measurements': {}}
    machine['benchmarks'] = benchmarks
    benchmarks['kernels'] = {
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

    USAGE_FACTOR = 0.5

    cores = list(range(1, machine['cores per socket']+1))
    for mem in machine['memory hierarchy']:
        measurement = {}
        machine['benchmarks']['measurements'][mem['level']] = measurement

        for threads_per_core in range(1, machine['threads per core']+1):
            threads = [c*threads_per_core for c in cores]
            if mem['size per group'] is not None:
                total_sizes = [
                    max(int(mem['size per group'])*c/mem['cores per group'],
                        int(mem['size per group']))*USAGE_FACTOR
                    for c in cores]
            else:
                last_mem = machine['memory hierarchy'][-2]
                total_sizes = [float(last_mem['size per group'])/USAGE_FACTOR for c in cores]
            sizes_per_core = [t/cores[i] for i, t in enumerate(total_sizes)]
            sizes_per_thread = [t/threads[i] for i, t in enumerate(total_sizes)]

            measurement[threads_per_core] = {
                'threads per core': threads_per_core,
                'cores': copy(cores),
                'threads': threads,
                'size per core': sizes_per_core,
                'size per thread': sizes_per_thread,
                'total size': total_sizes,
                'results': {}, }

    print('Progress: ', end='', file=sys.stderr)
    sys.stderr.flush()
    for mem_level in list(machine['benchmarks']['measurements'].keys()):
        for threads_per_core in list(machine['benchmarks']['measurements'][mem_level].keys()):
            measurement = machine['benchmarks']['measurements'][mem_level][threads_per_core]
            measurement['results'] = {}
            for kernel in list(machine['benchmarks']['kernels'].keys()):
                measurement['results'][kernel] = []
                for i, total_size in enumerate(measurement['total size']):
                    measurement['results'][kernel].append(measure_bw(
                        kernel,
                        int(float(total_size)/1000),
                        threads_per_core,
                        machine['threads per core'],
                        measurement['cores'][i],
                        sockets=1))

                    print('.', end='', file=sys.stderr)
                    sys.stderr.flush()

    print(yaml.dump(machine))

if __name__ == '__main__':
    main()
