#!/usr/bin/env python
from __future__ import print_function

import subprocess
import re
import sys
from pprint import pprint
from itertools import chain
from copy import copy

import yaml

from prefixedunit import PrefixedUnit

def get_match_or_break(regex, haystack, flags=re.MULTILINE):
    m = re.search(regex, haystack, flags)
    assert m, "could not find "+repr(regex)
    return m.groups()

def get_machine_topology():
    topo = subprocess.Popen(['likwid-topology'], stdout=subprocess.PIPE).communicate()[0]
    machine = {
        'model name': get_match_or_break(r'^CPU type:\s+(.+?)\s*$', topo)[0],
        'sockets': int(get_match_or_break(r'^Sockets:\s+([0-9]+)\s*$', topo)[0]),
        'cores per socket': 
            int(get_match_or_break(r'^Cores per socket:\s+([0-9]+)\s*$', topo)[0]),
        'threads per core':
            int(get_match_or_break(r'^Threads per core:\s+([0-9]+)\s*$', topo)[0]),
        'clock': 'INFORMATION_REQUIRED',
        'FLOPs per cycle': 'INFORMATION_REQUIRED',
        'micro-architecture': 'INFORMATION_REQUIRED',
        'icc architecture flags': 'INFORMATION_REQUIRED',
        'cacheline size': 'INFORMATION_REQUIRED',
        'overlapping ports': 'INFORAMTION_REQUIRED',
        'non-overlapping ports': 'INFORMATION_REQUIRED',
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
            mem_level['level'] = 'L'+line.split(':')[1].strip()
        elif line.startswith('Size:'):
            mem_level['size per group'] = PrefixedUnit(line.split(':')[1].strip())
        elif line.startswith('Cache groups:'):
            mem_level['groups'] = line.count('(')
            mem_level['cores per group'] =  (machine['cores per socket'] *
                 machine['sockets']) / mem_level['groups']
            mem_level['threads per group'] = \
                mem_level['cores per group'] * machine['threads per core']
        mem_level['cycles per cacheline transfer'] = 'INFORMATION_REQUIRED'
        mem_level['bandwidth per core'] = 'INFORMATION_REQUIRED'
        mem_level['max. total bandwidth'] = 'INFORMATION_REQUIRED'
        
        if len(mem_level) == 8:
            machine['memory hierarchy'].append(mem_level)
            mem_level = {}
    machine['memory hierarchy'].append({
        'level': 'MEM',
        'size per group': None,
        'cores per group': machine['cores per socket'],
        'threads per group': machine['threads per core'] * machine['cores per socket'],
        'cycles per cacheline transfer': None,
        'bandwidth': 'INFORMATION_REQUIRED'
    })
    
    return machine

def measure_bw(type_, total_size, threads_per_core, max_threads_per_core, cores_per_socket, sockets, iterations=1000):
    """*size* is given in kilo bytes"""
    groups = []
    for s in range(sockets):
        groups += ['-w',
             'S'+str(s)+':'+str(total_size)+'kB:'+str(threads_per_core*cores_per_socket)+':1:'+str(int(max_threads_per_core/threads_per_core))]
    # for older likwid versions add ['-g', str(sockets), '-i', str(iterations)] to cmd
    cmd = ['likwid-bench', '-t', type_]+groups
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()[0]
    bw = float(get_match_or_break(r'^MByte/s:\s+([0-9]+(?:\.[0-9]+)?)\s*$', output)[0])
    return PrefixedUnit(bw, 'MB/s')

def main():
    machine = get_machine_topology()
    
    total_threads = machine['threads per core'] * machine['cores per socket']
    benchmarks = {'kernels': {}, 'measurements': {}}
    machine['benchmarks'] = benchmarks
    benchmarks['kernels'] = {
        'load': {
            'read streams': {'streams': 1, 'bytes': PrefixedUnit(8,'B')},
            'read+write streams': {'streams': 0, 'bytes': PrefixedUnit(0,'B')},
            'write streams': {'streams': 0, 'bytes': PrefixedUnit(0,'B')},
            'FLOPs per iteration': 0},
        'copy': {
            'read streams': {'streams': 1, 'bytes': PrefixedUnit(8,'B')},
            'read+write streams': {'streams': 0, 'bytes': PrefixedUnit(0,'B')},
            'write streams': {'streams': 1, 'bytes': PrefixedUnit(8,'B')},
            'FLOPs per iteration': 0},
        'update': {
            'read streams': {'streams': 1, 'bytes': PrefixedUnit(8,'B')},
            'read+write streams': {'streams': 1, 'bytes': PrefixedUnit(8,'B')},
            'write streams': {'streams': 1, 'bytes': PrefixedUnit(8,'B')},
            'FLOPs per iteration': 0},
        'triad': {
            'read streams': {'streams': 3, 'bytes': PrefixedUnit(24,'B')},
            'read+write streams': {'streams': 1, 'bytes': PrefixedUnit(8,'B')},
            'write streams': {'streams': 1, 'bytes': PrefixedUnit(8,'B')},
            'FLOPs per iteration': 2},
        'daxpy': {
            'read streams': {'streams': 2, 'bytes': PrefixedUnit(16,'B')},
            'read+write streams': {'streams': 1, 'bytes': PrefixedUnit(8,'B')},
            'write streams': {'streams': 1, 'bytes': PrefixedUnit(8,'B')},
            'FLOPs per iteration': 2},}
    
    USAGE_FACTOR = 0.5
    
    cores = range(1, machine['cores per socket']+1)
    for mem in machine['memory hierarchy']:
        measurement = {}
        machine['benchmarks']['measurements'][mem['level']] = measurement
        
        for threads_per_core in range(1, machine['threads per core']+1):
            threads = [c*threads_per_core for c in cores]
            if mem['size per group'] is not None:
                total_sizes = [
                    max(mem['size per group']*c/mem['cores per group'],
                        mem['size per group'])*USAGE_FACTOR
                    for c in cores]
            else:
                last_mem = machine['memory hierarchy'][-2]
                total_sizes = [
                    max(last_mem['size per group']*c/mem['cores per group'],
                        last_mem['size per group'])/USAGE_FACTOR
                    for c in cores]
            sizes_per_core = [t/cores[i] for i, t in enumerate(total_sizes)]
            sizes_per_thread = [t/threads[i] for i, t in enumerate(total_sizes)]
            
            measurement[threads_per_core] = {
                'threads per core': threads_per_core,
                'cores': copy(cores),
                'threads': threads,
                'size per core': sizes_per_core,
                'size per thread': sizes_per_thread,
                'total size': total_sizes,
                'results': {},}
    
    print('Progress: ', end='', file=sys.stderr)
    sys.stderr.flush()
    for mem_level in machine['benchmarks']['measurements'].keys():
        for threads_per_core in machine['benchmarks']['measurements'][mem_level].keys():
            measurement = machine['benchmarks']['measurements'][mem_level][threads_per_core]
            measurement['results'] = {}
            for kernel in machine['benchmarks']['kernels'].keys():
                measurement['results'][kernel] = []
                for i, total_size in enumerate(measurement['total size']):
                    measurement['results'][kernel].append(measure_bw(
                        kernel,
                        int(float(total_size)/1000),
                        threads_per_core,
                        machine['threads per core'],
                        measurement['cores'][i],
                        1,  # Sockets
                        iterations=1000))
    
                    print('.', end='', file=sys.stderr)
                    sys.stderr.flush()
    
    print(yaml.dump(machine))
           
if __name__ == '__main__':
    main()
