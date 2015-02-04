#!/usr/bin/env python
from __future__ import print_function

import subprocess
import re
import sys
from pprint import pprint

import yaml

class UnitValue:
    PREFIXES = {'k': 1e3, 'M': 1e6, 'G': 1e9, 'T': 1e13, 'P': 1e16, 'E':1e19, 'Z': 1e21, 'Y': 1e24,
                '': 1}
    
    def __init__(self, *args):
        args = list(args)
        
        if len(args) == 1:
            if type(args[0]) == str:
                m = re.match(r'^(?P<value>[0-9]+(?:\.[0-9]+)?)\s+(?P<prefix>[kMGTP])?(?P<unit>.+)$', args[0])
                assert m, "Could not parse unit paramtere "+repr(s)
                g = m.groups()
                args = [float(g[0]), g[1], g[2]]
            else:
                args = [float(args[0]),'','']
        
        if len(args) == 2:
            m = re.match(r'^(?P<prefix>[kMGTP])?(?P<unit>.+)$', args[1])
            assert m, "Could not parse unit parameter"+repr(args)
            gd = m.groupdict()
            args = [float(args[0]), gd['prefix'], gd['unit']]
        
        if args[1] is None:
            args[1] = ''
        
        assert args[1] in self.PREFIXES, "Unknown prefix: "+repr(args[1])
        self.value, self.prefix, self.unit = args
    
    def base_value(self):
        '''gives value without prefix'''
        return self.value*self.PREFIXES[self.prefix]
    
    def good_prefix(self, max_error=0.01, round_length=2, min_prefix='', max_prefix=None):
        '''
        returns the largest prefix where the relative error is bellow *max_error* although rounded
        by *round_length*
        
        if *max_prefix* is found in UnitValue.PREFIXES, returned value will not exceed this prefix.
        if *min_prefix* is given, returned value will atleast be of that prefix (no matter the 
        error)
        '''
        good_prefix = min_prefix
        base_value = self.base_value()
    
        for k,v in self.PREFIXES.items():
            # Ignoring to large prefixes
            if max_prefix is not None and v > self.PREFIXES[max_prefix]:
                continue
        
            # Check that differences is < relative error *max_error*
            if abs(round(base_value/v, round_length)*v - base_value) > base_value*max_error:
                continue
        
            # Check that resulting number is >= 0.9
            if abs(round(base_value/v, round_length)) < 0.9:
                continue
        
            # Check if prefix is larger then already chosen
            if v < self.PREFIXES[good_prefix]:
                continue
        
            # seems to be okay
            good_prefix = k
    
        return good_prefix
        
    def with_prefix(self, prefix):
        return self.__class__(
            self.base_value()/self.PREFIXES[prefix], prefix, self.unit)
    
    def __str__(self):
        good_prefix = self.good_prefix()
        if self.prefix == good_prefix:
            return '{0:.2f} {1}{2}'.format(self.value, self.prefix, self.unit)
        else:
            return self.with_prefix(good_prefix).__str__()
    
    def __repr__(self):
        return '{0}({1!r}, {2!r}, {3!r})'.format(
            self.__class__.__name__, self.value, self.prefix, self.unit)

def get_match_or_break(regex, haystack, flags=re.MULTILINE):
    m = re.search(regex, haystack, flags)
    assert m, "could not find "+repr(regex)
    return m.groups()

def get_machine_topology():
    topo = subprocess.Popen(['likwid-topology'], stdout=subprocess.PIPE).communicate()[0]
    machine = {
        'sockets': int(get_match_or_break(r'^Sockets:\s+([0-9]+)\s*$', topo)[0]),
        'cores per socket': int(get_match_or_break(r'^Cores per socket:\s+([0-9]+)\s*$', topo)[0]),
        'threads per core': int(get_match_or_break(r'^Threads per core:\s+([0-9]+)\s*$', topo)[0])
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
    machine['cache stack'] = []
    cache = {}
    for line in topo[cache_start:cache_end].split('\n'):
        if line.startswith('Level:'):
            cache['level'] = int(line.split(':')[1].strip())
        elif line.startswith('Size:'):
            cache['size'] = UnitValue(line.split(':')[1].strip())
        elif line.startswith('Cache groups:'):
            cache['groups'] = line.count('(')
            cache['threads per group'] = (machine['threads per core'] * 
                machine['cores per socket'] * machine['sockets']) / cache['groups']
        if len(cache) == 4:
            machine['cache stack'].append(cache)
            cache = {}
    
    return machine

def measure_bw(type_, total_size, threads_per_socket, sockets, iterations=1000):
    """*size* is given in kilo bytes"""
    groups = []
    for s in range(sockets):
        groups += ['-w',
             'S'+str(s)+':'+str(total_size)+'kB:'+str(threads_per_socket)]
    # for older likwid versions add ['-g', str(sockets), '-i', str(iterations)] to cmd
    cmd = ['likwid-bench', '-t', type_]+groups
    #print(cmd, end='')
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()[0]
    bw = float(get_match_or_break(r'^MByte/s:\s+([0-9]+(?:\.[0-9]+)?)\s*$', output)[0])
    #print(UnitValue(bw, 'MB/s'))
    return UnitValue(bw, 'MB/s')

if __name__ == '__main__':
    machine = get_machine_topology()
    pprint(machine)
    
    total_threads = machine['threads per core'] * machine['cores per socket']
    measurement = {}
    measurement_setup = {
        'testcases': {
            'load': {'read streams': 1, 'write streams': 0, 'FLOPs': 0},
            'copy': {'read streams': 1, 'write streams': 1, 'FLOPs': 0},
            'update': {'read streams': 1, 'write streams': 1, 'FLOPs': 0},
            'triad': {'read streams': 3, 'write streams': 1, 'FLOPs': 2},
            'daxpy': {'read streams': 2, 'write streams': 1, 'FLOPs': 2}},
        'threads': range(
            machine['threads per core'], total_threads+1, machine['threads per core']),
        'cache utilization': 0.75,
        'data sizes': {},
    }
    
    for cache in machine['cache stack']:
        measurement_setup['data sizes'][cache['level']] = {}
        for threads in measurement_setup['threads']:
            total_size = max(cache['size'].base_value()*threads/cache['threads per group'],
                             cache['size'].base_value())
            per_thread = total_size/threads
            measurement_setup['data sizes'][cache['level']][threads] = \
                {'total': UnitValue(measurement_setup['cache utilization']*total_size, 'B'), 
                 'per thread': UnitValue(measurement_setup['cache utilization']*per_thread, 'B')}
    measurement_setup['data sizes']['MEM'] = {}
    for threads in measurement_setup['threads']:
        last_level_cache = machine['cache stack'][-1]
        total_size = max(
            last_level_cache['size'].base_value()*threads/last_level_cache['threads per group'],
            cache['size'].base_value())*4
        per_thread = total_size/threads
        measurement_setup['data sizes']['MEM'][threads] = \
            {'total': UnitValue(total_size, 'B'), 
             'per thread': UnitValue(per_thread, 'B')}
    pprint(measurement_setup)
    
    print('Progress: ', end='')
    sys.stdout.flush()
    for testcase in measurement_setup['testcases'].keys():
        measurement[testcase] = {}
        for level, thread_sizes in measurement_setup['data sizes'].items():
            measurement[testcase][level] = {}
            for threads, size in thread_sizes.items():
                # TODO scale to multisocket
                #print({'threads': threads, 
                #       'cache level': level, 
                #       'total size': str(size['total'][0])+'kB'})
                measurement[testcase][level][threads] = measure_bw(
                    testcase,
                    total_size=size['total'].with_prefix('k').value,
                    threads_per_socket=threads,
                    sockets=1,
                    iterations=int(5e6/size['total'].with_prefix('k').value))
                print('.', end='')
                sys.stdout.flush()
                #break
    print()
    pprint(measurement)
    
    out = dict(machine=machine, measurement_setup=measurement_setup, measurement=measurement)
    open('machine.yaml', 'w').write(yaml.dump(out))
    
    for testcase in measurement:
        print('-'*len(testcase)+'\n'+testcase+'\n'+'-'*len(testcase))
        print('Level', '|'.join(map(
            lambda i: '{0:^12}'.format(i), measurement[testcase][1].keys()))+'|', sep='|')
        for cache_level in measurement[testcase]:
            print('{0:>5}|'.format(cache_level), end='')
            for thread_count, result in measurement[testcase][cache_level].items():
                print('{0!s:>12}|'.format(result), end='', sep='|')
            print()
           
    