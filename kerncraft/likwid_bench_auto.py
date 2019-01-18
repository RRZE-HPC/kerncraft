#!/usr/bin/env python3
import collections
import sys
import subprocess
import re
from copy import copy
from pprint import pprint

from ruamel import yaml

from .prefixedunit import PrefixedUnit
from . import __version__


def get_match_or_break(regex, haystack, flags=re.MULTILINE):
    m = re.search(regex, haystack, flags)
    if not m:
        raise ValueError("could not find " + repr(regex) + " in " + repr(haystack))
    return m.groups()


def get_likwid_topology(cmd: str='likwid-topology') -> str:
    try:
        topo = subprocess.Popen(
            ['likwid-topology'], stdout=subprocess.PIPE
        ).communicate()[0].decode("utf-8")
    except OSError as e:
        print('likwid-topology execution failed ({}), is it installed and loaded?'.format(e),
              file=sys.stderr)
        sys.exit(1)
    return topo


def read_cpuinfo(cpuinfo_path: str='/proc/cpuinfo') -> str:
    with open(cpuinfo_path, 'r') as f:
        cpuinfo = f.read()
    return cpuinfo


def get_machine_topology(cpuinfo_path: str='/proc/cpuinfo') -> dict:
    topo = get_likwid_topology()
    cpuinfo = read_cpuinfo(cpuinfo_path)

    sockets = int(get_match_or_break(r'^Sockets:\s+([0-9]+)\s*$', topo)[0])
    cores_per_socket = int(get_match_or_break(r'^Cores per socket:\s+([0-9]+)\s*$', topo)[0])
    numa_domains_per_socket = \
        int(get_match_or_break(r'^NUMA domains:\s+([0-9]+)\s*$', topo)[0]) // sockets
    cores_per_numa_domain = cores_per_socket // numa_domains_per_socket
    machine = {
        'kerncraft version': __version__,
        'model type': get_match_or_break(r'^CPU type:\s+(.+?)\s*$', topo)[0],
        'model name': get_match_or_break(r'^model name\s+:\s+(.+?)\s*$', cpuinfo)[0],
        'sockets': sockets,
        'cores per socket': cores_per_socket,
        'threads per core': int(get_match_or_break(r'^Threads per core:\s+([0-9]+)\s*$', topo)[0]),
        'NUMA domains per socket': numa_domains_per_socket,
        'cores per NUMA domain': cores_per_numa_domain,
        'clock': 'INFORMATION_REQUIRED (e.g., 2.7 GHz)',
        'FLOPs per cycle': {'SP': {'total': 'INFORMATION_REQUIRED',
                                   'FMA': 'INFORMATION_REQUIRED',
                                   'ADD': 'INFORMATION_REQUIRED',
                                   'MUL': 'INFORMATION_REQUIRED'},
                            'DP': {'total': 'INFORMATION_REQUIRED',
                                   'FMA': 'INFORMATION_REQUIRED',
                                   'ADD': 'INFORMATION_REQUIRED',
                                   'MUL': 'INFORMATION_REQUIRED'}},
        'micro-architecture-modeler': 'INFORMATION_REQUIRED (options: OSACA, IACA)',
        'micro-architecture': 'INFORMATION_REQUIRED (options: NHM, WSM, SNB, IVB, HSW, BDW, SKL, SKX)',
        # TODO retrive flags automatically from compiler with -march=native
        'compiler': collections.OrderedDict([
                    ('icc', 'INFORMATION_REQUIRED (e.g., -O3 -fno-alias -xAVX)'),
                    ('clang', 'INFORMATION_REQUIRED (e.g., -O3 -mavx, -D_POSIX_C_SOURCE=200112L'),
                    ('gcc', 'INFORMATION_REQUIRED (e.g., -O3 -march=ivybridge)')]),
        'cacheline size': 'INFORMATION_REQUIRED (in bytes, e.g. 64 B)',
        'overlapping model': {
            'ports': 'INFORAMTION_REQUIRED (list of ports as they appear in IACA, e.g.)'
                     ', ["0", "0DV", "1", "2", "2D", "3", "3D", "4", "5", "6", "7"])',
            'performance counter metric':
                'INFORAMTION_REQUIRED Example:'
                'max(UOPS_DISPATCHED_PORT_PORT_0__PMC2, UOPS_DISPATCHED_PORT_PORT_1__PMC3,'
                '    UOPS_DISPATCHED_PORT_PORT_4__PMC0, UOPS_DISPATCHED_PORT_PORT_5__PMC1)'
        },
        'non-overlapping model': {
            'ports': 'INFORAMTION_REQUIRED (list of ports as they appear in IACA, e.g.)'
                     ', ["0", "0DV", "1", "2", "2D", "3", "3D", "4", "5", "6", "7"])',
            'performance counter metric':
                'INFORAMTION_REQUIRED Example:'
                'max(UOPS_DISPATCHED_PORT_PORT_0__PMC2, UOPS_DISPATCHED_PORT_PORT_1__PMC3,'
                '    UOPS_DISPATCHED_PORT_PORT_4__PMC0, UOPS_DISPATCHED_PORT_PORT_5__PMC1)'
        }
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
            mem_level = {'level': 'L' + line.split(':')[1].strip()}
            machine['memory hierarchy'].append(mem_level)
            if mem_level['level'] != 'L1':
                mem_level['non-overlap upstream throughput'] = [
                    'INFORMATION_REQUIRED (e.g. 24 B/cy)',
                    'INFORMATION_REQUIRED (e.g. "half-duplex" or "full-duplex")']
        elif line.startswith('Size:'):
            size = PrefixedUnit(line.split(':')[1].strip())
            mem_level['cache per group'] = {
                'sets': 'INFORMATION_REQUIRED (sets*ways*cl_size=' + str(size) + ')',
                'ways': 'INFORMATION_REQUIRED (sets*ways*cl_size=' + str(size) + ')',
                'cl_size': 'INFORMATION_REQUIRED (sets*ways*cl_size=' + str(size) + ')',
                'replacement_policy': 'INFORMATION_REQUIRED (options: LRU, FIFO, MRU, RR)',
                'write_allocate': 'INFORMATION_REQUIRED (True/False)',
                'write_back': 'INFORMATION_REQUIRED (True/False)',
                'load_from': 'L' + str(int(mem_level['level'][1:]) + 1),
                'store_to': 'L' + str(int(mem_level['level'][1:]) + 1)}
            mem_level['size per group'] = size
        elif line.startswith('Cache groups:'):
            mem_level['groups'] = line.count('(')
            mem_level['cores per group'] = \
                (machine['cores per socket'] * machine['sockets']) // mem_level['groups']
            mem_level['threads per group'] = \
                int(mem_level['cores per group'] * machine['threads per core'])
        mem_level['performance counter metrics'] = {
            'accesses': 'INFORMATION_REQUIRED (e.g., L1D_REPLACEMENT__PMC0)',
            'misses': 'INFORMATION_REQUIRED (e.g., L2_LINES_IN_ALL__PMC1)',
            'evicts': 'INFORMATION_REQUIRED (e.g., L2_LINES_OUT_DIRTY_ALL__PMC2)'
        }

    # Remove last caches load_from and store_to:
    del machine['memory hierarchy'][-1]['cache per group']['load_from']
    del machine['memory hierarchy'][-1]['cache per group']['store_to']

    machine['memory hierarchy'].append({
        'level': 'MEM',
        'cores per group': int(machine['cores per socket']),
        'threads per group': int(machine['threads per core'] * machine['cores per socket']),
        'non-overlap upstream throughput':
            ['full socket memory bandwidth',
             'INFORMATION_REQUIRED (e.g. "half-duplex" or "full-duplex")'],
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
            ':1:' + str(int(max_threads_per_core / threads_per_core))]
    # for older likwid versions add ['-g', str(sockets), '-i', str(iterations)] to cmd
    cmd = ['likwid-bench', '-t', type_] + groups
    sys.stderr.write(' '.join(cmd))
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()[0].decode('utf-8')
    if not output:
        print(' '.join(cmd) + ' returned no output, possibly wrong version installed '
                              '(requires 4.0 or later)', file=sys.stderr)
        sys.exit(1)
    bw = float(get_match_or_break(r'^MByte/s:\s+([0-9]+(?:\.[0-9]+)?)\s*$', output)[0])
    print(' ', PrefixedUnit(bw, 'MB/s'), file=sys.stderr)
    return PrefixedUnit(bw, 'MB/s')


def cli():
    # TODO support everything described here
    if '-h' in sys.argv[1:] or '--help' in sys.argv[1:]:
        print("""Usage:', sys.argv[0], '[-h] {collect|measure} [machinefile] | upgrade machinefile

        collect will retrieve as much hardware information as possible, without benchmarking
        measure will do the same as collect, but also include memory benchmarks

        If machinefile already exists the CPU name will be compared. If they matche, measurements
        will proceed and the file is updated accordingly. All other information in the file
        (typically manually inserted) will be left alone.

        If no machinefile is given, the information will be printed to stdout.

        upgrade will transform machinefile to the most up-to-date machine file version.
        """)


def main():
    machine = get_machine_topology()

    machine['benchmarks'] = {
        'kernels': {
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
                'FLOPs per iteration': 2}, },
        'measurements': {}}

    USAGE_FACTOR = 0.66
    MEM_FACTOR = 15.0

    cores = list(range(1, machine['cores per socket'] + 1))
    for mem in machine['memory hierarchy']:
        measurement = {}
        machine['benchmarks']['measurements'][mem['level']] = measurement

        for threads_per_core in range(1, machine['threads per core'] + 1):
            threads = [c * threads_per_core for c in cores]
            if mem['size per group'] is not None:
                total_sizes = [
                    PrefixedUnit(max(int(mem['size per group']) * c / mem['cores per group'],
                                     int(mem['size per group'])) * USAGE_FACTOR, 'B')
                    for c in cores]
            else:
                last_mem = machine['memory hierarchy'][-2]
                total_sizes = [last_mem['size per group'] * MEM_FACTOR for c in cores]
            sizes_per_core = [t / cores[i] for i, t in enumerate(total_sizes)]
            sizes_per_thread = [t / threads[i] for i, t in enumerate(total_sizes)]

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
                        int(float(total_size) / 1000),
                        threads_per_core,
                        machine['threads per core'],
                        measurement['cores'][i],
                        sockets=1))

                    print('.', end='', file=sys.stderr)
                    sys.stderr.flush()

    print(yaml.dump(machine, Dumper=yaml.Dumper))


if __name__ == '__main__':
    main()
