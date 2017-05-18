from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import

import subprocess
from functools import reduce
import operator
import sys
from distutils.spawn import find_executable
from pprint import pprint
import re

import six

from kerncraft.kernel import KernelCode


class Benchmark(object):
    """
    this will produce a benchmarkable binary to be used with likwid
    """

    name = "benchmark"

    @classmethod
    def configure_arggroup(cls, parser):
        pass

    def __init__(self, kernel, machine, args=None, parser=None):
        """
        *kernel* is a Kernel object
        *machine* describes the machine (cpu, cache and memory) characteristics
        *args* (optional) are the parsed arguments from the comand line
        """
        if not isinstance(kernel, KernelCode):
            raise ValueError("Kernel was not derived from code, can not perform Benchmark "
                             "analysis.")
        self.kernel = kernel
        self.machine = machine
        self._args = args
        self._parser = parser

        if args:
            # handle CLI info
            pass

    def perfctr(self, cmd, group='MEM', cpu='S0:0', code_markers=True, pin=True):
        '''
        runs *cmd* with likwid-perfctr and returns result as dict
        
        *group* may be a performance group known to likwid-perfctr or an event string.
        Only works with single core!
        '''

        # Making sure iaca.sh is available:
        if find_executable('likwid-perfctr') is None:
            print("likwid-perfctr was not found. Make sure likwid is installed and found in PATH.",
                  file=sys.stderr)
            sys.exit(1)

        # FIXME currently only single core measurements support!
        perf_cmd = ['likwid-perfctr', '-f', '-O', '-g', group]

        if pin:
            perf_cmd += ['-C', cpu]
        else:
            perf_cmd += ['-c', cpu]

        if code_markers:
            perf_cmd.append('-m')

        perf_cmd += cmd
        if self._args.verbose > 1:
            print(' '.join(perf_cmd))
        try:
            output = subprocess.check_output(perf_cmd).decode('utf-8').split('\n')
        except subprocess.CalledProcessError as e:
            print("Executing benchmark failed: {!s}".format(e), file=sys.stderr)
            sys.exit(1)

        results = {}
        ignore = True
        for l in output:
            l = l.split(',')
            try:
                # Metrics
                results[l[0]] = float(l[1])
            except:
                pass
            try:
                # Event counters
                counter_value = int(l[2])
                if re.fullmatch(r'[A-Z0-9_]+', l[0]) and re.fullmatch(r'[A-Z0-9]+', l[1]):
                    results.setdefault(l[0], {})
                    results[l[0]][l[1]] = counter_value
            except (IndexError, ValueError):
                pass

        return results

    def analyze(self):
        bench = self.kernel.build(verbose=self._args.verbose > 1)

        # Build arguments to pass to command:
        args = [bench] + [six.text_type(s) for s in list(self.kernel.constants.values())]

        # Determan base runtime with 100 iterations
        runtime = 0.0
        time_per_repetition = 0.2/10.0

        while runtime < 0.15:
            # Interpolate to a 0.2s run
            if time_per_repetition != 0.0:
                repetitions = 0.2//time_per_repetition
            else:
                repetitions *= 10

            result = self.perfctr(args+[six.text_type(repetitions)], group="MEM")
            runtime = result['Runtime (RDTSC) [s]']
            time_per_repetition = runtime/float(repetitions)
        results = {'MEM': result}
        
        # Gather remaining counters counters
        if self.machine['micro-architecture'] in ['IVB', 'HSW', 'BDW']:
            event_counters = {'nOL': [('UOPS_DISPATCHED_PORT_PORT_0', 'PMC0'),
                                      ('UOPS_DISPATCHED_PORT_PORT_1', 'PMC1'),
                                      ('UOPS_DISPATCHED_PORT_PORT_4', 'PMC2'),
                                      ('UOPS_DISPATCHED_PORT_PORT_5', 'PMC3')],
                              'OL': [('MEM_UOPS_RETIRED_LOADS', 'PMC0'),
                                     ('MEM_UOPS_RETIRED_STORES', 'PMC1')],
                              'L2L3': [('L1D_REPLACEMENT', 'PMC0'),
                                       ('L1D_M_EVICT', 'PMC1'),
                                       ('L2_LINES_IN_ALL', 'PMC2'),
                                       ('L2_LINES_OUT_DIRTY_ALL', 'PMC3')]}
        else:
            event_counters = {}
        
        for group_name, ctrs in event_counters.items():
            ctrs = ','.join([':'.join(c) for c in ctrs])
            results[group_name] = self.perfctr(args+[six.text_type(repetitions)], group=ctrs)

        self.results = {'raw output': results}
        
        # Build phenomenological ECM model:
        if self.machine['micro-architecture'] in ['IVB', 'HSW', 'BDW']:
            element_size = self.kernel.datatypes_size[self.kernel.datatype]
            elements_per_cacheline = float(self.machine['cacheline size']) // element_size
            total_iterations = self.kernel.iteration_length() * repetitions
            total_cachelines = total_iterations/elements_per_cacheline
            self.results['ECM'] = {
                'T_nOL': max(results['nOL']['UOPS_DISPATCHED_PORT_PORT_0']['PMC0'],
                             results['nOL']['UOPS_DISPATCHED_PORT_PORT_1']['PMC1'],
                             results['nOL']['UOPS_DISPATCHED_PORT_PORT_4']['PMC2'],
                             results['nOL']['UOPS_DISPATCHED_PORT_PORT_5']['PMC3'])
                         / total_cachelines,
                # TODO check for AVX,SSE,.. loads to determin cy/uop, currently assuming 1cy/uop
                'T_OL': results['OL']['MEM_UOPS_RETIRED_LOADS']['PMC0']
                        / total_cachelines,
                'T_L1L2': (results['L2L3']['L1D_REPLACEMENT']['PMC0'] +
                           results['L2L3']['L1D_M_EVICT']['PMC1'])
                          * 2.0 # two cycles per CL
                          / total_cachelines,
                'T_L2L3': (results['L2L3']['L2_LINES_IN_ALL']['PMC2'] +
                           results['L2L3']['L2_LINES_OUT_DIRTY_ALL']['PMC3'])
                          * 2.0 # two cycles per CL
                          / total_cachelines,
                'T_L3MEM': results['MEM']['Memory data volume [GBytes]']*1e9
                           /  (40e9/float(self.machine['clock'])) # 40GB/s / GHz = B/cy
                           / total_cachelines
            }
        else:
            self.results['ECM'] = None
            

        self.results['Runtime (per repetition) [s]'] = time_per_repetition
        # TODO make more generic to support other (and multiple) constantnames
        # TODO support SP (devide by 4 instead of 8.0)
        iterations_per_repetition = reduce(
            operator.mul,
            [self.kernel.subs_consts(max_-min_)/self.kernel.subs_consts(step)
             for idx, min_, max_, step in self.kernel._loop_stack],
            1)
        self.results['Iterations per repetition'] = iterations_per_repetition
        iterations_per_cacheline = float(self.machine['cacheline size'])/8.0
        cys_per_repetition = time_per_repetition*float(self.machine['clock'])
        self.results['Runtime (per cacheline update) [cy/CL]'] = \
            (cys_per_repetition/iterations_per_repetition)*iterations_per_cacheline
        self.results['MEM volume (per repetition) [B]'] = \
            results['MEM']['Memory data volume [GBytes]']*1e9/repetitions
        self.results['Performance [MFLOP/s]'] = \
            sum(self.kernel._flops.values())/(time_per_repetition/iterations_per_repetition)/1e6
        if 'Memory bandwidth [MBytes/s]' in results['MEM']:
            self.results['MEM BW [MByte/s]'] = results['MEM']['Memory bandwidth [MBytes/s]']
        else:
            self.results['MEM BW [MByte/s]'] = results['MEM']['Memory BW [MBytes/s]']
        self.results['Performance [MLUP/s]'] = (iterations_per_repetition/time_per_repetition)/1e6
        self.results['Performance [MIt/s]'] = (iterations_per_repetition/time_per_repetition)/1e6

    def report(self, output_file=sys.stdout):
        if self._args.verbose > 0:
            print('Runtime (per repetition): {:.2g} s'.format(
                      self.results['Runtime (per repetition) [s]']),
                  file=output_file)
        if self._args.verbose > 0:
            print('Iterations per repetition: {!s}'.format(
                     self.results['Iterations per repetition']),
                  file=output_file)
        print('Runtime (per cacheline update): {:.2f} cy/CL'.format(
                  self.results['Runtime (per cacheline update) [cy/CL]']),
              file=output_file)
        print('MEM volume (per repetition): {:.0f} Byte'.format(
                  self.results['MEM volume (per repetition) [B]']),
              file=output_file)
        print('Performance: {:.2f} MFLOP/s'.format(self.results['Performance [MFLOP/s]']),
              file=output_file)
        print('Performance: {:.2f} MLUP/s'.format(self.results['Performance [MLUP/s]']),
              file=output_file)
        print('Performance: {:.2f} It/s'.format(self.results['Performance [MIt/s]']),
              file=output_file)
        if self._args.verbose > 0:
            print('MEM bandwidth: {:.2f} MByte/s'.format(self.results['MEM BW [MByte/s]']),
                  file=output_file)
        print('', file=output_file)
        
        if self.results['ECM']:
            print('Phenomenological ECM model: {{ {T_OL:.1f} || {T_nOL:.1f} | {T_L1L2:.1f} | '
                  '{T_L2L3:.1f} | {T_L3MEM:.1f} }} cy/CL'.format(
                **self.results['ECM']))
            print('T_OL assumes that only 1 Load per cycle may be retiered, which is not true for '
                  'SSE loads on SNB, IVY, HSW and BDW.')
