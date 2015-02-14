#!/usr/bin/env python

from __future__ import print_function
from textwrap import dedent
from pprint import pprint
from functools import reduce
import operator
import math
import copy
import sys
import subprocess
import re
import imp

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot_support = True
except ImportError:
    plot_support = False

from kerncraft import intervals
from kerncraft.kernel import Kernel
from kerncraft.machinemodel import MachineModel

# Datatype sizes in bytes
datatype_size = {'double': 8, 'float': 4}


def blocking(indices, block_size, initial_boundary=0):
    '''
    splits list of integers into blocks of block_size. returns block indices.
    
    first block element is located at initial_boundary (default 0).
    
    >>> blocking([0, -1, -2, -3, -4, -5, -6, -7, -8, -9], 8)
    [0,-1]
    >>> blocking([0], 8)
    [0]
    '''
    blocks = []
    
    for idx in indices:
        bl_idx = (idx-initial_boundary)/block_size
        if bl_idx not in blocks:
            blocks.append(bl_idx)
    blocks.sort()
    
    return blocks


def find(f, seq):
  """Return first item in sequence where f(item) == True."""
  for item in seq:
    if f(item): 
      return item
    

def flatten_dict(d):
    '''
    transforms 2d-dict d[i][k] into a new 1d-dict e[(i,k)] with 2-tuple keys
    '''
    e = {}
    for k in d.keys():
        for l in d[k].keys():
            e[(k,l)] = d[k][l]
    return e


class ArrayAccess:
    '''
    This class under stands array acceess based on multidimensional indices and one dimensional
    functions of style i*N+j
    '''
    def __init__(self, aref=None, array_info=None):
        '''If *aref* (and *array_info*) is present, this will be parsed.'''
        self.sym_expr = None
        self.index_parameters = {}
        if aref:
            self.parse(aref, array_info=array_info)
    
    def parse(self, aref, dim=0, array_info=None):
        if type(aref.name) is c_ast.ArrayRef:
            from_higher_dim = self.parse(aref.name, dim=dim+1, array_info=array_info)
        else:
            from_higher_dim = 0
        
        if array_info and dim != 0:
            dim_stride = reduce(lambda m,n: sympy.Mul(m, n, evaluate=False), array_info[1][:dim])
        else:
            dim_stride = 1
        
        sym_expr = sympy.Mul(conv_ast_to_sympy(aref.subscript), dim_stride, evaluate=False) + \
            from_higher_dim
        
        if dim == 0:
            # Store gathered information
            self.sym_expr = sym_expr
        else:
            # Return gathered information
            return sym_expr
    
    def extract_parameters(self):
        eq = self.sym_expr.simplify()
        terms = eq.as_ordered_terms()
        
        for t in terms:
            if type(t) is sympy.Symbol:
                self.index_parameters[t.name] = {}
            elif type(t) is sympy.Mul:
                for a in t.args:
                    if type(a) is sympy.Symbol:
                        self.index_parameters[a.name] = {}
                    elif type(a) is sympy.Integer:
                        #self.
                        pass
    
    def __repr__(self):
        return unicode(self.sym_expr)


class ECMData:
    """
    class representation of the Execution-Cache-Memory Model (only the data part)

    more info to follow...
    """
    
    name = "Execution-Cache-Memory (data transfers only)"
    
    @classmethod
    def configure_arggroup(cls, parser):
        pass
    
    def __init__(self, kernel, machine, args=None):
        """
        *kernel* is a Kernel object
        *machine* describes the machine (cpu, cache and memory) characteristics
        *args* (optional) are the parsed arguments from the comand line
        """
        self.kernel = kernel
        self.machine = machine
        self._args = args
        
        if args:
            # handle CLI info
            pass
    
    def _calculate_relative_offset(self, name, access_dimensions):
        '''
        returns the offset from the iteration center in number of elements and the order of indices
        used in access.
        '''
        offset = 0
        base_dims = self.kernel._variables[name][1]
        
        for dim, offset_info in enumerate(access_dimensions):
            offset_type, idx_name, dim_offset = offset_info
            assert offset_type == 'rel', 'Only relative access to arrays is supported at the moment'
            
            if offset_type == 'rel':
                offset += dim_offset*reduce(operator.mul, base_dims[dim+1:], 1)
            else:
                # should not happen
                pass
        
        return offset
    
    def _calculate_iteration_offset(self, name, index_order, loop_index):
        '''
        returns the offset from one to the next iteration using *loop_index*.
        *index_order* is the order used by the access dimensions e.g. 'ijk' corresponse to [i][j][k]
        *loop_index* specifies the loop to be used for iterations (this is typically the inner 
        moste one)
        '''
        offset = 0
        base_dims = self.kernel._variables[name][1]
            
        for dim, index_name in enumerate(index_order):
            if loop_index == index_name:
                offset += reduce(operator.mul, base_dims[dim+1:], 1)
        
        return offset
    
    def _get_index_order(self, access_dimensions):
        '''Returns the order of indices used in *access_dimensions*.'''
        return ''.join(map(lambda d: d[1], access_dimensions))
    
    def _expand_to_cacheline_blocks(self, first, last):
        '''
        Returns first and last values wich align with cacheline blocks, by increasing range.
        '''
        # TODO how to handle multiple datatypes (with different size)?
        element_size = datatype_size['double']
        elements_per_cacheline = int(float(self.machine['cacheline size'])) / element_size
        
        first = first - first%elements_per_cacheline
        last = last - last%elements_per_cacheline + elements_per_cacheline - 1
        
        return [first, last]
    
    def calculate_cache_access(self):
        results = {}
        
        read_offsets = {var_name: dict() for var_name in self.kernel._variables.keys()}
        write_offsets = {var_name: dict() for var_name in self.kernel._variables.keys()}
        iteration_offsets = {var_name: dict() for var_name in self.kernel._variables.keys()}
        
        # TODO how to handle multiple datatypes (with different size)?
        element_size = datatype_size['double']
        elements_per_cacheline = int(float(self.machine['cacheline size'])) / element_size
        
        loop_order = ''.join(map(lambda l: l[0], self.kernel._loop_stack))
        
        for var_name in self.kernel._variables.keys():
            var_type, var_dims = self.kernel._variables[var_name]
            
            # Skip the following access: (they are hopefully kept in registers)
            #   - scalar values
            if var_dims is None: continue
            #   - access does not change with inner-most loop index
            writes = filter(lambda acs: loop_order[-1] in map(lambda a: a[1], acs),
                self.kernel._destinations.get(var_name, []))
            reads = filter(lambda acs: loop_order[-1] in map(lambda a: a[1], acs),
                self.kernel._sources.get(var_name, []))
            
            # Compile access pattern
            for r in reads:
                offset = self._calculate_relative_offset(var_name, r)
                idx_order = self._get_index_order(r)
                read_offsets[var_name].setdefault(idx_order, [])
                read_offsets[var_name][idx_order].append(offset)
            for w in writes:
                offset = self._calculate_relative_offset(var_name, w)
                idx_order = self._get_index_order(w)
                write_offsets[var_name].setdefault(idx_order, [])
                write_offsets[var_name][idx_order].append(offset)
            
            # Do unrolling so that one iteration equals one cacheline worth of workload:
            # unrolling is done on inner-most loop only!
            for i in range(1, elements_per_cacheline):
                for r in reads:
                    idx_order = self._get_index_order(r)
                    offset = self._calculate_relative_offset(var_name, r)
                    offset += i * self._calculate_iteration_offset(
                        var_name, idx_order, loop_order[-1])
                    read_offsets[var_name][idx_order].append(offset)
            
                    # Remove multiple access to same offsets
                    read_offsets[var_name][idx_order] = \
                        sorted(list(set(read_offsets[var_name][idx_order])), reverse=True)
                
                for w in writes:
                    idx_order = self._get_index_order(w)
                    offset = self._calculate_relative_offset(var_name, w)
                    offset += i * self._calculate_iteration_offset(
                        var_name, idx_order, loop_order[-1])
                    write_offsets[var_name][idx_order].append(offset)
                    
                    # Remove multiple access to same offsets
                    write_offsets[var_name][idx_order] = \
                        sorted(list(set(write_offsets[var_name][idx_order])), reverse=True)
        
        # initialize misses and hits
        misses = {}
        hits = {}
        evicts = {}
        total_misses = {}
        total_hits = {}
        total_evicts = {}
        total_lines_misses = {}
        total_lines_hits = {}
        total_lines_evicts = {}
        
        self.results = {'memory hierarchy': [], 'cycles': []}
        
        # Check for layer condition towards all cache levels (except main memory/last level)
        for i, cache_info in list(enumerate(self.machine['memory hierarchy']))[:-1]:
            cache_size = int(float(cache_info['size per group']))
            cache_cycles = cache_info['cycles per cacheline transfer']
            
            trace_length = 0
            updated_length = True
            while updated_length:
                updated_length = False
                
                # Initialize cache, misses, hits and evicts for current level
                cache = {}
                misses[i] = {}
                hits[i] = {}
                evicts[i] = {}
                
                # We consider everythin a miss in the beginning
                # TODO here read and writes are treated the same, this implies write-allocate
                #      to support nontemporal stores, this needs to be changed
                for name in read_offsets.keys()+write_offsets.keys():
                    cache[name] = {}
                    misses[i][name] = {}
                    hits[i][name] = {}

                    for idx_order in read_offsets[name].keys()+write_offsets[name].keys():
                        cache[name][idx_order] = intervals.Intervals()
                        if i-1 not in misses:
                            misses[i][name][idx_order] = sorted(
                                read_offsets.get(name, {}).get(idx_order, []) + 
                                write_offsets.get(name, {}).get(idx_order, []),
                                reverse=True)
                        else:
                            misses[i][name][idx_order] = list(misses[i-1][name][idx_order])
                        hits[i][name][idx_order] = []
                
                # Caches are still empty (thus only misses)
                trace_count = 0
                cache_used_size = 0
                
                # Now we trace the cache access backwards (in time/iterations) and check for hits
                for var_name in misses[i].keys():
                    for idx_order in misses[i][var_name].keys():
                        iter_offset = self._calculate_iteration_offset(
                            var_name, idx_order, loop_order[-1])
                        
                        # Add cache trace
                        for offset in list(misses[i][var_name][idx_order]):
                            # If already present in cache add to hits
                            if offset in cache[var_name][idx_order]:
                                misses[i][var_name][idx_order].remove(offset)
                                
                                # We might have multiple hits on the same offset (e.g in DAXPY)
                                if offset not in hits[i][var_name][idx_order]:
                                    hits[i][var_name][idx_order].append(offset)
                                
                            # Add cache, we can do this since misses are sorted in reverse order of
                            # access and we assume LRU cache replacement policy
                            if iter_offset <= elements_per_cacheline:
                                # iterations overlap, thus we can savely add the whole range
                                cached_first, cached_last = self._expand_to_cacheline_blocks(
                                    offset-iter_offset*trace_length, offset+1)
                                cache[var_name][idx_order] &= intervals.Intervals(
                                    [cached_first, cached_last+1], sane=True)
                            else:
                                # There is no overlap, we can append the ranges onto one another
                                # TODO optimize this code section (and maybe merge with above)
                                new_cache = [self._expand_to_cacheline_blocks(o, o) for o in range(
                                    offset-iter_offset*trace_length, offset+1, iter_offset)]
                                new_cache = intervals.Intervals(*new_cache, sane=True)
                                cache[var_name][idx_order] &= new_cache
                                
                        trace_count += len(cache[var_name][idx_order]._data)
                        cache_used_size += len(cache[var_name][idx_order])*element_size
                
                # Calculate new possible trace_length according to free space in cache
                # TODO take CL blocked access into account
                # TODO make /2 customizable
                new_trace_length = trace_length + \
                    ((cache_size/2 - cache_used_size)/trace_count)/element_size
                
                if new_trace_length > trace_length:
                    trace_length = new_trace_length
                    updated_length = True
                
                # All writes to require the data to be evicted eventually
                evicts[i] = {var_name: dict() for var_name in self.kernel._variables.keys()}
                for name in write_offsets.keys():
                    for idx_order in write_offsets[name].keys():
                        evicts[i][name][idx_order] = list(write_offsets[name][idx_order])
            
            # Compiling stats
            total_misses[i] = sum(map(lambda l: sum(map(len, l.values())), misses[i].values()))
            total_hits[i] = sum(map(lambda l: sum(map(len, l.values())), hits[i].values()))
            total_evicts[i] = sum(map(lambda l: sum(map(len, l.values())), evicts[i].values()))
            
            total_lines_misses[i] = sum(map(
                lambda o: sum(map(lambda n: len(blocking(n, elements_per_cacheline)), o.values())),
                misses[i].values()))
            total_lines_hits[i] = sum(map(lambda o: sum(map(lambda n:
                len(blocking(n, elements_per_cacheline)), o.values())),
                hits[i].values()))
            total_lines_evicts[i] = sum(map(lambda o: sum(map(lambda n: 
                len(blocking(n,elements_per_cacheline)), o.values())),
                evicts[i].values()))
            
            if cache_cycles:
                # Cache
                cycles = (total_lines_misses[i]+total_lines_evicts[i]) * \
                     cache_cycles
            else:
                # Memory
                cycles = (total_lines_misses[i] + total_lines_evicts[i]) * \
                    elements_per_cacheline * element_size  * \
                    float(self.machine['clock']) / \
                    float(self.machine['memory hierarchy'][-2]['bandwidth'])
            
            self.results['memory hierarchy'].append({
                'index': i,
                'level': '{}'.format(cache_info['level']),
                'total misses': total_misses[i],
                'total hits': total_hits[i],
                'total evicts': total_evicts[i],
                'total lines misses': total_lines_misses[i],
                'total lines hits': total_lines_hits[i],
                'total lines evicts': total_lines_evicts[i],
                'trace length': trace_length,
                'misses': misses[i],
                'hits': hits[i],
                'evicts': evicts[i],
                'cycles': cycles})
            self.results['cycles'].append((
                '{}-{}'.format(cache_info['level'], self.machine['memory hierarchy'][i+1]['level']),
                cycles))
            
            # TODO remove the following by makeing testcases more versatile:
            self.results['{}-{}'.format(cache_info['level'], self.machine['memory hierarchy'][i+1]['level'])] = cycles
            
        return results

    def analyze(self):
        self._results = self.calculate_cache_access()
    
    def report(self):
        if self._args and self._args.verbose > 0:
            for r in self.results['memory hierarchy']:
                print('Trace legth per access in {}:'.format(r['level']), r['trace length'])
                print('Hits in {}:'.format(r['level']), r['total hits'], r['hits'])
                print('Misses in {}: {} ({}CL):'.format(
                    r['level'], r['total misses'], r['total lines misses']),
                    r['misses'])
                print('Evicts from {} {} ({}CL):'.format(
                    r['level'], r['total evicts'], r['total lines evicts']),
                    r['evicts'])
        
        for level, cycles in self.results['cycles']:
            print('{} = {}cy'.format(level, cycles))


class ECMCPU:
    """
    class representation of the Execution-Cache-Memory Model (only the operation part)

    more info to follow...
    """
    
    name = "Execution-Cache-Memory (CPU operations only)"
    
    @classmethod
    def configure_arggroup(cls, parser):
        parser.add_argument('--asm-block', metavar='BLOCK', default='auto',
                            help='Number of ASM block to mark for IACA, "auto" for automatic ' + \
                                 'selection or "manual" for interactiv selection.')
    
    def __init__(self, kernel, machine, args=None):
        """
        *kernel* is a Kernel object
        *machine* describes the machine (cpu, cache and memory) characteristics
        *args* (optional) are the parsed arguments from the comand line
        """
        self.kernel = kernel
        self.machine = machine
        self._args = args
        
        if args:
            # handle CLI info
            if self._args.asm_block not in ['auto', 'manual']:
                try:
                    self._args.asm_block = int(args.asm_block)
                except ValueError:
                    parser.error('--asm-block can only be "auto", "manual" or an integer')
    
    def analyze(self):
        # For the IACA/CPU analysis we need to compile and assemble
        asm_name = self.kernel.compile(compiler_args=self.machine['icc architecture flags'])
        bin_name = self.kernel.assemble(
            asm_name, iaca_markers=True, asm_block=self._args.asm_block)
        
        iaca_output = subprocess.check_output(
            ['iaca.sh', '-64', '-arch', self.machine['micro-architecture'], bin_name])

        # Get total cycles per loop iteration
        match = re.search(
            r'^Block Throughput: ([0-9\.]+) Cycles', iaca_output, re.MULTILINE)
        assert match, "Could not find Block Throughput in IACA output."
        block_throughput = float(match.groups()[0])
        
        # Find ports and cyles per port
        ports = filter(lambda l: l.startswith('|  Port  |'), iaca_output.split('\n'))
        cycles = filter(lambda l: l.startswith('| Cycles |'), iaca_output.split('\n'))
        assert ports and cycles, "Could not find ports/cylces lines in IACA output."
        ports = map(str.strip, ports[0].split('|'))[2:]
        cycles = map(str.strip, cycles[0].split('|'))[2:]
        port_cycles = []
        for i in range(len(ports)):
            if '-' in ports[i] and ' ' in cycles[i]:
                subports = map(str.strip, ports[i].split('-'))
                subcycles = filter(bool, cycles[i].split(' '))
                port_cycles.append((subports[0], float(subcycles[0])))
                port_cycles.append((subports[0]+subports[1], float(subcycles[1])))
            elif ports[i] and cycles[i]:
                port_cycles.append((ports[i], float(cycles[i])))
        port_cycles = dict(port_cycles)
        
        match = re.search(r'^Total Num Of Uops: ([0-9]+)', iaca_output, re.MULTILINE)
        assert match, "Could not find Uops in IACA output."
        uops = float(match.groups()[0])
        
        # Normalize to cycles per cacheline
        block_elements = self.kernel.blocks[self.kernel.block_idx][1]['loop_increment']
        block_size = block_elements*8  # TODO support SP
        block_to_cl_ratio = float(self.machine['cacheline size'])/block_size
        
        port_cycles = dict(map(lambda i: (i[0], i[1]*block_to_cl_ratio), port_cycles.items()))
        uops = uops*block_to_cl_ratio
        block_throughput = block_throughput*block_to_cl_ratio
        
        # Compile most relevant information
        T_OL = max([v for k,v in port_cycles.items() if k in self.machine['overlapping ports']])
        T_nOL = max(
            [v for k,v in port_cycles.items() if k in self.machine['non-overlapping ports']])
        
        # Create result dictionary
        self.results = {
            'port cycles': port_cycles,
            'block throughput': block_throughput,
            'uops': uops,
            'T_nOL': T_nOL, 
            'T_OL': T_OL}
        
    def report(self):
        if self._args and self._args.verbose > 0:
            print('Ports and cycles:', self.results['port cycles'])
            print('Uops:', self.results['uops'])
            
            print('Throughput: {}cy per CL'.format(self.results['block throughput']))
        
        print('T_nOL = {}cy'.format(self.results['T_nOL']))
        print('T_OL = {}cy'.format(self.results['T_OL']))


class ECM:
    """
    class representation of the Execution-Cache-Memory Model (data and operations)

    more info to follow...
    """
    
    name = "Execution-Cache-Memory"
    
    @classmethod
    def configure_arggroup(cls, parser):
        # they are being configured in ECMData and ECMCPU
        parser.add_argument('--ecm-plot',
            help='Filename to save ECM plot to (supported extensions: pdf, png, svg and eps)')
    
    def __init__(self, kernel, machine, args=None):
        """
        *kernel* is a Kernel object
        *machine* describes the machine (cpu, cache and memory) characteristics
        *args* (optional) are the parsed arguments from the comand line
        """
        self.kernel = kernel
        self.machine = machine
        self._args = args
        
        if args:
            # handle CLI info
            pass
        
        self._CPU = ECMCPU(kernel, machine, args)
        self._data = ECMData(kernel, machine, args)
    
    def analyze(self):
        self._CPU.analyze()
        self._data.analyze()
        self.results = copy.deepcopy(self._CPU.results)
        self.results.update(copy.deepcopy(self._data.results))
    
    def report(self):
        report = ''
        if self._args and self._args.verbose > 0:
            self._CPU.report()
            self._data.report()
        
        report += '{{ {} || {} | {} }}'.format(
            self.results['T_OL'],
            self.results['T_nOL'],
            ' | '.join([str(i[1]) for i in self.results['cycles']]))
        
        print(report)
        
        if self._args and self._args.ecm_plot:
            assert plot_support, "matplotlib couldn't be imported. Plotting is not supported."
            
            results = self.results['cycles'] + [('T_OL', self.results['T_OL']) +
                                               ('T_nOL', self.results['T_nOL'])]
            
            fig = plt.figure(frameon=False)
            ax = fig.add_subplot(1,1,1)

            sorted_overlapping_ports = sorted(
                map(lambda p: (p, self.results['port cycles'][p]),
                    self.machine['overlapping ports']),
                key=lambda x: x[1])

            yticks_labels = []
            yticks = []
            xticks_labels = []
            xticks = []
            
            # Plot configuration
            height = 0.8

            i = 0
            # T_OL
            colors = [(254./255, 177./255., 178./255.)] + [(255./255., 255./255., 255./255.)] * \
                 (len(sorted_overlapping_ports) - 1)
            for p, c in sorted_overlapping_ports:
                ax.barh(i, c, height, align='center', color=colors.pop())
                yticks_labels.append(p)
                yticks.append(i)
                i += 1
            xticks.append(sorted_overlapping_ports[-1][1])
            xticks_labels.append('{:.1f}'.format(sorted_overlapping_ports[-1][1]))

            # T_nOL + memory transfers
            y = 0
            colors = [(187./255., 255/255., 188./255.)] * (len(self.results['cycles'])) + \
                [(119./255, 194./255., 255./255.)]
            for k,v in [('T_nOL', self.results['T_nOL'])]+self.results['cycles']:
                ax.barh(i, v, height, y, align='center', color=colors.pop())
                ax.text(y+v/2.0, i, k, ha='center', va='center')
                xticks.append(y+v)
                xticks_labels.append('{:.1f}'.format(y+v))
                y += v
            yticks_labels.append('LD')
            yticks.append(i)

            ax.tick_params(axis='y', which='both', left='off', right='off')
            ax.tick_params(axis='x', which='both', top='off')
            ax.set_xlabel('t [cy]')
            ax.set_ylabel('execution port')
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticks_labels)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks_labels)
            ax.grid(axis='x', alpha=0.7, linestyle='--')
            fig.savefig(self._args.ecm_plot)
