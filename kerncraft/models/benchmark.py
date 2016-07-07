from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import

import subprocess
from functools import reduce
import operator
import sys
import six
from distutils.spawn import find_executable

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
        '''

        # Making sure iaca.sh is available:
        if find_executable('likwid-perfctr') is None:
            print("likwid-perfctr was not found. Make sure likwid is installed and found in PATH.",
                  file=sys.stderr)
            sys.exit(1)
        
        # FIXME currently only single core measurements support!
        perf_cmd = ['likwid-perfctr', '-O', '-g', group]

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
            if ignore and (l.startswith('Event,core 0') or l.startswith('Metric,Core 0')):
                ignore = False
            elif ignore or not l:
                continue

            l = l.split(',')
            results[l[0]] = l[1:]

        return results

    def analyze(self):
        bench = self.kernel.build(self.machine['compiler'],
                                  cflags=self.machine['compiler flags'],
                                  verbose=self._args.verbose > 1)

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

            result = self.perfctr(args+[six.text_type(repetitions)])
            runtime = float(result['Runtime (RDTSC) [s]'][0])
            time_per_repetition = runtime/float(repetitions)

        self.results = {'raw output': result}

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
            float(result['Memory data volume [GBytes]'][0])*1e9/repetitions
        self.results['Performance [MFLOP/s]'] = \
            sum(self.kernel._flops.values())/(time_per_repetition/iterations_per_repetition)/1e6
        if 'Memory bandwidth [MBytes/s]' in result:
            self.results['MEM BW [MByte/s]'] = float(result['Memory bandwidth [MBytes/s]'][0])
        else:
            self.results['MEM BW [MByte/s]'] = float(result['Memory BW [MBytes/s]'][0])
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
        print('MEM volume (per repetition): {:.2f} Byte'.format(
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
