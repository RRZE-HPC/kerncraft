#!/usr/bin/env python3
"""Stand-alone Benchmark model and helper functions."""
import os
import subprocess
from functools import reduce
import operator
import sys
from distutils.spawn import find_executable
import re
from collections import defaultdict
import string
import pprint
import contextlib

from math import ceil

from kerncraft.prefixedunit import PrefixedUnit
from .base import PerformanceModel

from .benchmark import sympy_safe_key, pprint_nosort, fix_env_variable, group_iterator, \
    register_options, eventstr, build_minimal_runs, get_supported_likwid_groups


class StandaloneBenchmark(PerformanceModel):
    """Run a likwid performance counter measurement on an existing binary."""

    name = "stand-alone benchmark"

    default_region = ''

    @classmethod
    def configure_arggroup(cls, parser):
        """Configure argument parser."""
        parser.add_argument(
            '--no-phenoecm', action='store_true',
            help='Disables the phenomenological ECM model building.')
        parser.add_argument(
            '--ignore-warnings', action='store_true',
            help='Ignore warnings about mismatched CPU model and frequency.')

    def __init__(self, kernel, machine, args=None, parser=None, no_phenoecm=False, verbose=0):
        """
        Create Benchmark model from kernel and machine objects.

        *kernel* is a Kernel object
        *machine* describes the machine (cpu, cache and memory) characteristics
        *args* (optional) are the parsed arguments from the command line

        If *args* is None, *no_phenoecm* and *verbose* are used.
        """
        self.kernel = kernel
        self.machine = machine
        self._args = args
        self._parser = parser
        self.results = {}

        self.benchmarked_regions = set()

        if args:
            self.no_phenoecm = args.no_phenoecm
            self.verbose = args.verbose
        else:
            self.no_phenoecm = no_phenoecm
            self.verbose = verbose

        self.iterations = 10

        if self._args.cores > 1 and not self.no_phenoecm:
            print("Info: phenological ECM model can only be created with a single core benchmark.")
            self.no_phenoecm = True
        elif not self.no_phenoecm:
            print("Info: If this takes too long and a phenological ECM model is not required, run "
                  "with --no-phenoecm.", file=sys.stderr)

        if not self.machine.current_system(print_diff=True):
            print("WARNING: current machine and machine description do not match.")
            if not args.ignore_warnings:
                print("You may ignore warnings by adding --ignore-warnings to the command line.")
                sys.exit(1)

    def perfctr(self, cmd, group='MEM', code_markers=True):
        """
        Run *cmd* with likwid-perfctr and returns result as dict.

        *group* may be a performance group known to likwid-perfctr or an event string.

        if CLI argument cores > 1, running with multi-core, otherwise single-core
        """
        # Making sure likwid-perfctr is available:
        if find_executable('likwid-perfctr') is None:
            print("likwid-perfctr was not found. Make sure likwid is installed and found in PATH.",
                  file=sys.stderr)
            sys.exit(1)

        # FIXME currently only single core measurements support!
        perf_cmd = ['likwid-perfctr', '-f', '-O', '-g', group]

        cpu = 'S0:0'
        if self._args.cores > 1:
            cpu += '-'+str(self._args.cores-1)

        # Pinned and measured on cpu
        perf_cmd += ['-C', cpu]

        # benchmark only marked regions when --marker is specified
        if self._args.marker['use_marker']:
            perf_cmd.append('-m')

        perf_cmd += cmd
        if self.verbose > 1:
            print(' '.join(perf_cmd))
        try:
            with fix_env_variable('OMP_NUM_THREADS', None):
                output = subprocess.check_output(perf_cmd).decode('utf-8').split('\n')
        except subprocess.CalledProcessError as e:
            print("Executing benchmark failed: {!s}".format(e), file=sys.stderr)
            sys.exit(1)

        # TODO multicore output is different and needs to be considered here!
        results = defaultdict(dict)
        current_region = self.default_region
        for line in output:
            line = line.split(',')

            if line[0] == 'TABLE' and line[1].startswith("Region "):
                # new likwid region
                current_region = line[1].split(' ', 1)[1]
                self.benchmarked_regions.add(current_region)

            try:
                # Metrics
                results[current_region][line[0]] = float(line[1])
                continue
            except ValueError:
                # Would not convert to float
                pass
            except IndexError:
                # Not a parable line (did not contain any commas)
                continue
            try:
                # Event counters
                if line[2] == '-' or line[2] == 'nan':
                    counter_value = 0
                else:
                    counter_value = int(line[2])
                if re.fullmatch(r'[A-Z0-9_]+', line[0]) and re.fullmatch(r'[A-Z0-9]+', line[1]):
                    results[current_region].setdefault(line[0], {})
                    results[current_region][line[0]][line[1]] = counter_value
                    continue
            except (IndexError, ValueError):
                pass

            # benchmarked with likwid markers
            if line[0] == 'call count':
                results[current_region]['call count'] = line[1] if not current_region == self.default_region else None

            if line[0].endswith(":") and len(line) == 3 and line[2] == "":
                # CPU information strings
                results[self.default_region][line[0]] = line[1]
                continue

        # if no region was specified, use ''
        if not self.benchmarked_regions:
            self.benchmarked_regions.add('')

        # Check that frequency during measurement matches machine description
        expected_clock = float(self.machine['clock'])
        current_clock = float(results[self.default_region]['CPU clock:'].replace(" GHz", "")) * 1e9
        if abs(current_clock - expected_clock) > expected_clock * 0.01:
            print("WARNING: measured CPU frequency and machine description did "
                  "not match during likwid-perfctr run. ({!r} vs {!r})".format(
                expected_clock, current_clock))
            if not self._args.ignore_warnings:
                print("You may ignore warnings by adding --ignore-warnings to the command line.")
                sys.exit(1)

        return results


    def adjust_variables(self, runtimes):
        """
        Adjusts variables in order to extrapolate to a 1.5s run. 
        The first variables defined will be adjusted first.
        """

        def adjust_loop(varname, region):
            for i in range(len(self.kernel.loops[region])):
                if self.kernel.loops[region][i]['variable'] == varname:
                    self.kernel.loops[region][i]['end'] = self.kernel.define[varname]['value'] - \
                        self.kernel.loops[region][i]['offset']

        adjustable = False

        # get regions where runtime < 1.5
        regions = []
        for regionname, runtimeval in runtimes.items():
            if runtimeval < 1.5:
                regions.append(regionname)

        # loop through
        for region in regions:
            for name, variable in self.kernel.define.items():
                if variable['adjustable'] and region in variable['region']:
                    factor = ceil(2.0 / runtimes[region])
                    if variable['scale'] == 'lin' and variable['value'] < variable['stop']:
                        self.kernel.define[name]['value'] = min(variable['value'] * factor,
                                                                variable['stop'])
                        self.kernel.set_constant(name, self.kernel.define[name]['value'])
                        adjust_loop(name, region)
                        adjustable = True
                        break
                    elif variable['scale'] == 'log' and variable['value'] > variable['stop']:
                        self.kernel.define[name]['value'] = max(
                            variable['value'] * (10 ** (-factor)),variable['stop'])
                        self.kernel.set_constant(name, self.kernel.define[name]['value'])
                        adjust_loop(name, region)
                        adjustable = True
                        break

        return adjustable


    def analyze(self, output_file=sys.stdout):
        """Run analysis."""
        bench = self.kernel.binary
        element_size = self.kernel.datatypes_size[self.kernel.datatype]

        # Build arguments to pass to command:
        input_args = []

        # Determine base runtime with 10 iterations
        runtimes = {'': 0.0}
        time_per_repetition = None

        repetitions = self.kernel.repetitions

        results = defaultdict(dict)

        # TODO if cores > 1, results are for openmp run. Things might need to be changed here!

        # Check for MEM group existence
        valid_groups = get_supported_likwid_groups()
        if "MEM" in valid_groups:
            group = "MEM"
        else:
            group = valid_groups[0]

        while min(runtimes.values()) < 1.5:

            if min(runtimes.values()) == 0.0:
                adjustable = True
            else:
                adjustable = self.adjust_variables(runtimes)

            if not adjustable:
                print("WARNING: Could not extrapolate to a 1.5s run (for at least one region). "
                      "Measurements might not be accurate.", file=output_file)
                break

            input_args = [str(variable['value']) for variable in self.kernel.define.values()]

            results = self.perfctr([bench] + input_args, group=group)

            if not self.kernel.regions:
                # no region specified for --marker -> benchmark all
                self.kernel.regions = set(results.keys())
                if len(self.kernel.regions) > 1:
                    self.kernel.regions.discard('')
                self.kernel.check()
                repetitions = self.kernel.repetitions

            else:
                # check if specified region(s) are found in results
                for region in self.kernel.regions:
                    if not region in results.keys():
                        print("Region '{}' was not found in the likwid output.".format(region),
                              file=output_file)
                        sys.exit(-1)

            runtimes = dict(zip(self.kernel.regions,
                                [results[r]['Runtime (RDTSC) [s]'] for r in self.kernel.regions]))

            for region in self.kernel.regions:
                if self.kernel.repetitions[region]['marker']:
                    repetitions[region]['value'] = results[region]['call count']
                elif self.kernel.repetitions[region]['variable']:
                    repetitions[region]['value'] = \
                        self.kernel.define[self.kernel.repetitions[region]['variable']]['value']
                elif self.kernel.repetitions[region]['value']:
                    repetitions[region]['value'] = self.kernel.repetitions[region]['value']

            time_per_repetition = {r: runtimes[r] / float(repetitions[r]['value'])
                                   for r in self.kernel.regions}
        raw_results_collection = [results]

        # repetitions were obtained from likwid marker and time per repetition is too small
        # -> overhead introduced by likwid markers is not negligible
        for region in self.kernel.regions:
            if self.kernel.repetitions[region]['marker']:
                # repetitions were obtained from likwid markers
                if time_per_repetition[region] < 1.0:
                    # time per repetition is <1000 ms (overhead is not negligible)
                    print("WARNING: Overhead introduced by likwid markers for region {} might not "
                          "be negligible (usage of '-R marker').\n".format(region),
                          file=output_file)


        if self.benchmarked_regions - self.kernel.regions:
            print('WARNING: following likwid regions were found but not specified to be analysed:\n'
                  '{}'.format(self.benchmarked_regions - self.kernel.regions))

        # Base metrics for further metric computations:


        # collect counters for phenoecm run
        if not self.no_phenoecm:
            # Build events and sympy expressions for all model metrics
            T_OL, event_counters = self.machine.parse_perfmetric(
                self.machine['overlapping model']['performance counter metric'])
            T_data, event_dict = self.machine.parse_perfmetric(
                self.machine['non-overlapping model']['performance counter metric'])
            event_counters.update(event_dict)
            cache_metrics = defaultdict(dict)
            for i in range(len(self.machine['memory hierarchy']) - 1):
                cache_info = self.machine['memory hierarchy'][i]
                name = cache_info['level']
                for k, v in cache_info['performance counter metrics'].items():
                    cache_metrics[name][k], event_dict = self.machine.parse_perfmetric(v)
                    event_counters.update(event_dict)

            # Compile minimal runs to gather all required events
            minimal_runs = build_minimal_runs(list(event_counters.values()))
            measured_ctrs = {}

            for region in self.kernel.regions:
                measured_ctrs[region] = {}

            for run in minimal_runs:
                ctrs = ','.join([eventstr(e) for e in run])
                r = self.perfctr([bench] + input_args, group=ctrs)
                raw_results_collection.append(r)

                for region in self.kernel.regions:
                    measured_ctrs[region].update(r[region])


        # start analysing for each region
        for region in self.kernel.regions:

            raw_results = [r[region] for r in raw_results_collection]

            iterations_per_repetition = self.kernel.region__iterations_per_repetition(region)

            iterations_per_cacheline = (float(self.machine['cacheline size']) /
                                        self.kernel.region__bytes_per_iteration(region))
            cys_per_repetition = time_per_repetition[region] * float(self.machine['clock'])

            # Gather remaining counters
            if not self.no_phenoecm:

                # Match measured counters to symbols
                event_counter_results = {}
                for sym, ctr in event_counters.items():
                    event, regs, parameter = ctr[0], register_options(ctr[1]), ctr[2]
                    for r in regs:
                        if r in measured_ctrs[region][event]:
                            event_counter_results[sym] = measured_ctrs[region][event][r]

                # Analytical metrics needed for further calculation
                cl_size = float(self.machine['cacheline size'])
                total_iterations = iterations_per_repetition * repetitions[region]['value']
                total_cachelines = total_iterations / iterations_per_cacheline

                T_OL_result = T_OL.subs(event_counter_results) / total_cachelines
                cache_metric_results = defaultdict(dict)
                for cache, mtrcs in cache_metrics.items():
                    for m, e in mtrcs.items():
                        cache_metric_results[cache][m] = e.subs(event_counter_results)

                # Inter-cache transfers per CL
                cache_transfers_per_cl = {cache: {k: PrefixedUnit(v / total_cachelines, 'CL/CL')
                                                  for k, v in d.items()}
                                          for cache, d in cache_metric_results.items()}
                cache_transfers_per_cl['L1']['accesses'].unit = 'LOAD/CL'

                # Select appropriate bandwidth
                mem_bw, mem_bw_kernel = self.machine.get_bandwidth(
                    -1,  # mem
                    cache_metric_results['L3']['misses'],  # load_streams
                    cache_metric_results['L3']['evicts'],  # store_streams
                    1)

                data_transfers = {
                    # Assuming 0.5 cy / LOAD (SSE on SNB or IVB; AVX on HSW, BDW, SKL or SKX)
                    'T_nOL': (cache_metric_results['L1']['accesses'] / total_cachelines * 0.5),
                    'T_L1L2': ((cache_metric_results['L1']['misses'] +
                                cache_metric_results['L1']['evicts']) /
                               total_cachelines * cl_size /
                               self.machine['memory hierarchy'][1]['upstream throughput'][0]),
                    'T_L2L3': ((cache_metric_results['L2']['misses'] +
                                cache_metric_results['L2']['evicts']) /
                               total_cachelines * cl_size /
                               self.machine['memory hierarchy'][2]['upstream throughput'][0]),
                    'T_L3MEM': ((cache_metric_results['L3']['misses'] +
                                 cache_metric_results['L3']['evicts']) *
                                float(self.machine['cacheline size']) /
                                total_cachelines / mem_bw *
                                float(self.machine['clock']))
                }

                # Build phenomenological ECM model:
                ecm_model = {'T_OL': T_OL_result}
                ecm_model.update(data_transfers)
            else:
                event_counters = {}
                ecm_model = None
                cache_transfers_per_cl = None

            self.results[region] = {'raw output': raw_results, 'ECM': ecm_model,
                                    'data transfers': cache_transfers_per_cl,
                                    'Runtime (per repetition) [s]': time_per_repetition[region],
                                    'event counters': event_counters,
                                    'Iterations per repetition': iterations_per_repetition,
                                    'Iterations per cacheline': iterations_per_cacheline}

            self.results[region]['Runtime (per cacheline update) [cy/CL]'] = \
                (cys_per_repetition / iterations_per_repetition) * iterations_per_cacheline
            if 'Memory data volume [GBytes]' in results[region]:
                self.results[region]['MEM volume (per repetition) [B]'] = (
                    results[region]['Memory data volume [GBytes]'] * 1e9 /
                    repetitions[region]['value'])
            else:
                self.results[region]['MEM volume (per repetition) [B]'] = float('nan')
            self.results[region]['Performance [MFLOP/s]'] = ( self.kernel._flops[region] /
                (time_per_repetition[region] / iterations_per_repetition) / 1e6)
            if 'Memory bandwidth [MBytes/s]' in results[region]:
                self.results[region]['MEM BW [MByte/s]'] = \
                    results[region]['Memory bandwidth [MBytes/s]']
            elif 'Memory BW [MBytes/s]' in results[region]:
                self.results[region]['MEM BW [MByte/s]'] = results[region]['Memory BW [MBytes/s]']
            else:
                self.results[region]['MEM BW [MByte/s]'] = float('nan')
            self.results[region]['Performance [MLUP/s]'] = \
                (iterations_per_repetition / time_per_repetition[region]) / 1e6
            self.results[region]['Performance [MIt/s]'] = \
                (iterations_per_repetition / time_per_repetition[region]) / 1e6

    def report(self, output_file=sys.stdout):
        """Report gathered analysis data in human readable form."""

        for region in self.kernel.regions:

            print('\nResults for region \'{}\'\n'.format(region))

            if self.verbose > 1:
                with pprint_nosort():
                    pprint.pprint(self.results[region])

            if self.verbose > 0:
                print('Runtime (per repetition): {:.2g} s'.format(
                    self.results[region]['Runtime (per repetition) [s]']),
                    file=output_file)
            if self.verbose > 0:
                print('Iterations per repetition: {!s}'.format(
                    self.results[region]['Iterations per repetition']),
                    file=output_file)
            print('Runtime (per cacheline update): {:.2f} cy/CL'.format(
                self.results[region]['Runtime (per cacheline update) [cy/CL]']),
                file=output_file)
            print('MEM volume (per repetition): {:.0f} Byte'.format(
                self.results[region]['MEM volume (per repetition) [B]']),
                file=output_file)
            print(
                'Performance: {:.2f} MFLOP/s'.format(self.results[region]['Performance [MFLOP/s]']),
                file=output_file)
            print('Performance: {:.2f} MLUP/s'.format(self.results[region]['Performance [MLUP/s]']),
                  file=output_file)
            print('Performance: {:.2f} MIt/s'.format(self.results[region]['Performance [MIt/s]']),
                  file=output_file)
            if self.verbose > 0:
                print('MEM bandwidth: {:.2f} MByte/s'.format(
                    self.results[region]['MEM BW [MByte/s]']),
                      file=output_file)
            print(file=output_file)

            if not self.no_phenoecm:
                print("Data Transfers:")
                print("{:^8} |".format("cache"), end='')
                for metrics in self.results[region]['data transfers'].values():
                    for metric_name in sorted(metrics):
                        print(" {:^14}".format(metric_name), end='')
                    print()
                    break
                for cache, metrics in sorted(self.results[region]['data transfers'].items()):
                    print("{!s:^8} |".format(cache), end='')
                    for k, v in sorted(metrics.items()):
                        print(" {!s:^14}".format(v), end='')
                    print()
                print()

                print('Phenomenological ECM model: {{ {T_OL:.1f} || {T_nOL:.1f} | {T_L1L2:.1f} | '
                      '{T_L2L3:.1f} | {T_L3MEM:.1f} }} cy/CL'.format(
                    **{k: float(v) for k, v in self.results[region]['ECM'].items()}),
                    file=output_file)
                print('T_OL assumes that two loads per cycle may be retired, which is true for '
                      '128bit SSE/half-AVX loads on SNB and IVY, and 256bit full-AVX loads on HSW, '
                      'BDW, SKL and SKX, but it also depends on AGU availability.',
                      file=output_file)
