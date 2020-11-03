#!/usr/bin/env python3
"""Benchmark model and helper functions."""
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

from kerncraft.prefixedunit import PrefixedUnit
from .base import PerformanceModel


class sympy_safe_key:
    """Replacement for pprint._safe_key to be sympy-safe"""

    __slots__ = ['obj']

    def __init__(self, obj):
        self.obj = obj

    def __lt__(self, other):
        return ((str(type(self.obj)), id(self.obj)) < \
                (str(type(other.obj)), id(other.obj)))

@contextlib.contextmanager
def pprint_nosort():
    orig, pprint._safe_key = pprint._safe_key, sympy_safe_key
    try:
        yield
    finally:
        pprint._safe_key = orig


@contextlib.contextmanager
def fix_env_variable(name, value):
    """Fix environment variable to a value within context. Unset if value is None."""
    orig = os.environ.get(name, None)
    if value is not None:
        # Set if value is not None
        os.environ[name] = value
    elif name in os.environ:
        # Unset if value is None
        del os.environ[name]
    try:
        yield
    finally:
        if orig is not None:
            # Restore original value
            os.environ[name] = orig
        elif name in os.environ:
            # Unset
            del os.environ[name]


def group_iterator(group):
    """
    Iterate over simple regex-like groups.

    The only special character is a dash (-), which take the preceding and the following chars to
    compute a range. If the range is non-sensical (e.g., b-a) it will be empty

    Example:
    >>> list(group_iterator('a-f'))
    ['a', 'b', 'c', 'd', 'e', 'f']
    >>> list(group_iterator('148'))
    ['1', '4', '8']
    >>> list(group_iterator('7-9ab'))
    ['7', '8', '9', 'a', 'b']
    >>> list(group_iterator('0B-A1'))
    ['0', '1']
    """
    ordered_chars = string.ascii_letters + string.digits
    tokenizer = ('(?P<seq>[a-zA-Z0-9]-[a-zA-Z0-9])|'
                 '(?P<chr>.)')
    for m in re.finditer(tokenizer, group):
        if m.group('seq'):
            start, sep, end = m.group('seq')
            for i in range(ordered_chars.index(start), ordered_chars.index(end) + 1):
                yield ordered_chars[i]
        else:
            yield m.group('chr')


def register_options(regdescr):
    """
    Very reduced regular expressions for describing a group of registers.

    Only groups in square bracktes and unions with pipes (|) are supported.

    Examples:
    >>> list(register_options('PMC[0-3]'))
    ['PMC0', 'PMC1', 'PMC2', 'PMC3']
    >>> list(register_options('MBOX0C[01]'))
    ['MBOX0C0', 'MBOX0C1']
    >>> list(register_options('CBOX2C1'))
    ['CBOX2C1']
    >>> list(register_options('CBOX[0-3]C[01]'))
    ['CBOX0C0', 'CBOX0C1', 'CBOX1C0', 'CBOX1C1', 'CBOX2C0', 'CBOX2C1', 'CBOX3C0', 'CBOX3C1']
    >>> list(register_options('PMC[0-1]|PMC[23]'))
    ['PMC0', 'PMC1', 'PMC2', 'PMC3']
    """
    if not regdescr:
        yield None
    tokenizer = ('\[(?P<grp>[^]]+)\]|'
                 '(?P<chr>.)')
    for u in regdescr.split('|'):
        m = re.match(tokenizer, u)

        if m.group('grp'):
            current = group_iterator(m.group('grp'))
        else:
            current = [m.group('chr')]

        for c in current:
            if u[m.end():]:
                for r in register_options(u[m.end():]):
                    yield c + r
            else:
                yield c


def eventstr(event_tuple=None, event=None, register=None, parameters=None):
    """
    Return a LIKWID event string from an event tuple or keyword arguments.

    *event_tuple* may have two or three arguments: (event, register) or
    (event, register, parameters)

    Keyword arguments will be overwritten by *event_tuple*.

    >>> eventstr(('L1D_RE'+'PLACEMENT', 'PMC0', None))
    'L1D_RE'+'PLACEMENT:PMC0'
    >>> eventstr(('L1D_RE'+'PLACEMENT', 'PMC0'))
    'L1D_RE'+'PLACEMENT:PMC0'
    >>> eventstr(('MEM_UOPS_RETIRED_LOADS', 'PMC3', {'EDGEDETECT': None, 'THRESHOLD': 2342}))
    'MEM_UOPS_RETIRED_LOADS:PMC3:EDGEDETECT:THRESHOLD=0x926'
    >>> eventstr(event='DTLB_LOAD_MISSES_WALK_DURATION', register='PMC3')
    'DTLB_LOAD_MISSES_WALK_DURATION:PMC3'
    """
    if len(event_tuple) == 3:
        event, register, parameters = event_tuple
    elif len(event_tuple) == 2:
        event, register = event_tuple
    event_dscr = [event, register]

    if parameters:
        for k, v in sorted(event_tuple[2].items()):  # sorted for reproducability
            if type(v) is int:
                k += "={}".format(hex(v))
            event_dscr.append(k)
    return ":".join(event_dscr)


def build_minimal_runs(events):
    """Compile list of minimal runs for given events."""
    # Eliminate multiples
    events = [e for i, e in enumerate(events) if events.index(e) == i]

    # Build list of runs per register group
    scheduled_runs = {}
    scheduled_events = []
    cur_run = 0
    while len(scheduled_events) != len(events):
        for event_tpl in events:
            event, registers, parameters = event_tpl
            # Skip allready scheduled events
            if event_tpl in scheduled_events:
                continue
            # Compile explicit list of possible register locations
            for possible_reg in register_options(registers):
                # Schedule in current run, if register is not yet in use
                s = scheduled_runs.setdefault(cur_run, {})
                if possible_reg not in s:
                    s[possible_reg] = (event, possible_reg, parameters)
                    # ban from further scheduling attempts
                    scheduled_events.append(event_tpl)
                    break
        cur_run += 1

    # Collaps all register dicts to single runs
    runs = [list(v.values()) for v in scheduled_runs.values()]

    return runs


def get_supported_likwid_groups():
    """Return list of likwid groups, supported by current architecture and likwid version."""
    output = subprocess.check_output(['likwid-perfctr', '-a']).decode('utf-8')
    return re.findall('^\s*([A-Z_0-9]{2,})\s', output, re.MULTILINE)


class Benchmark(PerformanceModel):
    """Produce a benchmarkable binary to be used with likwid."""

    name = "benchmark"

    @classmethod
    def configure_arggroup(cls, parser):
        """Configure argument parser."""
        parser.add_argument(
            '--no-phenoecm', action='store_true',
            help='Disables the phenomenological ECM model building.')
        parser.add_argument(
            '--iterations', type=int, default=10,
            help='Number of outer-loop iterations (e.g. time loop) during benchmarking. '
                 'Default is 10, but actual number will be adapted to at least 0.2s runtime.')
        parser.add_argument(
            '--ignore-warnings', action='store_true',
            help='Ignore warnings about missmatched CPU model and frequency.')

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
        self.results = None

        if args:
            self.no_phenoecm = args.no_phenoecm
            self.verbose = args.verbose
            self.iterations = args.iterations
        else:
            self.no_phenoecm = no_phenoecm
            self.verbose = verbose
            self.iterations = 10

        if self._args.cores > 1 and not self.no_phenoecm:
            print("Info: phenological ECM model can only be created with a single core benchmark.",
                  file=sys.stderr)
            self.no_phenoecm = True
        elif "INFORAMTION_REQUIRED" in ''.join(
                [self.machine['overlapping model']['performance counter metric'],
                 self.machine['non-overlapping model']['performance counter metric']]):
            print("Info: disabled phenological ECM model, because definition is incomplete.",
                  file=sys.stderr)
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

        # code must be marked using likwid markers
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
        results = {}
        for line in output:
            line = line.split(',')
            try:
                # Metrics
                results[line[0]] = float(line[1])
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
                    results.setdefault(line[0], {})
                    results[line[0]][line[1]] = counter_value
                    continue
            except (IndexError, ValueError):
                pass
            if line[0].endswith(":") and len(line) == 3 and line[2] == "":
                # CPU information strings
                results[line[0]] = line[1]
                continue

        # Check that frequency during measurement matches machine description
        expected_clock = float(self.machine['clock'])
        if 'CPU clock:' in results:
            current_clock = float(results['CPU clock:'].replace(" GHz", "")) * 1e9
            if abs(current_clock - expected_clock) > expected_clock * 0.01:
                print("WARNING: measured CPU frequency and machine description did "
                      "not match during likwid-perfctr run. ({!r} vs {!r})".format(
                    expected_clock, current_clock))
                if not self._args.ignore_warnings:
                    print("You may ignore warnings by adding --ignore-warnings to the command line.")
                    sys.exit(1)

        return results

    def analyze(self):
        """Run analysis."""
        bench_filename, bench_lock_fp = self.kernel.build_executable(
            verbose=self.verbose > 1, openmp=self._args.cores > 1)
        element_size = self.kernel.datatypes_size[self.kernel.datatype]

        # Build arguments to pass to command:
        args = [str(s) for s in list(self.kernel.constants.values())]

        # Determine base runtime with 10 iterations
        runtime = 0.0
        time_per_repetition = 2.0 / 10.0
        repetitions = self.iterations // 10
        results = {}

        # TODO if cores > 1, results are for openmp run. Things might need to be changed here!

        # Check for MEM group existence
        valid_groups = get_supported_likwid_groups()
        if "MEM" in valid_groups:
            group = "MEM"
        else:
            group = valid_groups[0]

        while runtime < 1.5:
            # Interpolate to a 2.0s run
            if time_per_repetition != 0.0:
                repetitions = 2.0 // time_per_repetition
            else:
                repetitions = int(repetitions * 10)

            results = self.perfctr([bench_filename] + [str(repetitions)] + args, group=group)
            runtime = results['Runtime (RDTSC) [s]']
            time_per_repetition = runtime / float(repetitions)
        raw_results = [results]

        # Base metrics for further metric computations:
        # An iteration is equal to one high-level code inner-most-loop iteration
        iterations_per_repetition = reduce(
            operator.mul,
            [self.kernel.subs_consts(max_ - min_) / self.kernel.subs_consts(step)
             for idx, min_, max_, step in self.kernel._loop_stack],
            1)
        iterations_per_cacheline = (float(self.machine['cacheline size']) /
                                    self.kernel.bytes_per_iteration)
        cys_per_repetition = time_per_repetition * float(self.machine['clock'])

        try:
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
        except SyntaxError as e:
            print('Disabled Phenomenological ECM, due to syntax error in machine file '
                    'metrics:', e, file=sys.stderr)
            self.no_phenoecm = True

        # Gather remaining counters
        if self.no_phenoecm:
            bench_lock_fp.close()
            event_counters = {}
            ecm_model = None
            cache_transfers_per_cl = None
        else:
            # Compile minimal runs to gather all required events
            minimal_runs = build_minimal_runs(list(event_counters.values()))
            measured_ctrs = {}
            for run in minimal_runs:
                ctrs = ','.join([eventstr(e) for e in run])
                r = self.perfctr([bench_filename] + [str(repetitions)] + args, group=ctrs)
                raw_results.append(r)
                measured_ctrs.update(r)
            bench_lock_fp.close()
            # Match measured counters to symbols
            event_counter_results = {}
            for sym, ctr in event_counters.items():
                event, regs, parameter = ctr[0], register_options(ctr[1]), ctr[2]
                for r in regs:
                    if r in measured_ctrs[event]:
                        event_counter_results[sym] = measured_ctrs[event][r]

            # Analytical metrics needed for futher calculation
            cl_size = float(self.machine['cacheline size'])
            total_iterations = iterations_per_repetition * repetitions
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
                # TODO make this mapping generic
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

        self.results = {'raw output': raw_results, 'ECM': ecm_model,
                        'data transfers': cache_transfers_per_cl,
                        'Runtime (per repetition) [s]': time_per_repetition,
                        'event counters': event_counters,
                        'Iterations per repetition': iterations_per_repetition,
                        'Iterations per cacheline': iterations_per_cacheline}

        # TODO make more generic to support other (and multiple) constant names
        self.results['Runtime (per cacheline update) [cy/CL]'] = \
            (cys_per_repetition / iterations_per_repetition) * iterations_per_cacheline
        if 'Memory data volume [GBytes]' in results:
            self.results['MEM volume (per repetition) [B]'] = \
                results['Memory data volume [GBytes]'] * 1e9 / repetitions
        else:
            self.results['MEM volume (per repetition) [B]'] = float('nan')
        self.results['Performance [MFLOP/s]'] = \
            sum(self.kernel._flops.values()) / (
            time_per_repetition / iterations_per_repetition) / 1e6
        if 'Memory bandwidth [MBytes/s]' in results:
            self.results['MEM BW [MByte/s]'] = results['Memory bandwidth [MBytes/s]']
        elif 'Memory BW [MBytes/s]' in results:
            self.results['MEM BW [MByte/s]'] = results['Memory BW [MBytes/s]']
        else:
            self.results['MEM BW [MByte/s]'] = float('nan')
        self.results['Performance [MLUP/s]'] = \
            (iterations_per_repetition / time_per_repetition) / 1e6
        self.results['Performance [MIt/s]'] = \
            (iterations_per_repetition / time_per_repetition) / 1e6

    def report(self, output_file=sys.stdout):
        """Report gathered analysis data in human readable form."""
        if self.verbose > 1:
            with pprint_nosort():
                pprint.pprint(self.results)

        if self.verbose > 0:
            print('Runtime (per repetition): {:.2g} s'.format(
                self.results['Runtime (per repetition) [s]']),
                file=output_file)
        if self.verbose > 0:
            print('Iterations per repetition: {!s}'.format(
                self.results['Iterations per repetition']),
                file=output_file)
        print('Runtime (per cacheline update): {:.2f} cy/CL'.format(
            self.results['Runtime (per cacheline update) [cy/CL]']),
            file=output_file)
        print('MEM volume (per repetition): {:.0f} Byte'.format(
            self.results['MEM volume (per repetition) [B]']),
            file=output_file)
        print('Performance: {:.2f} MFLOP/s'.format(float(self.results['Performance [MFLOP/s]'])),
              file=output_file)
        print('Performance: {:.2f} MLUP/s'.format(float(self.results['Performance [MLUP/s]'])),
              file=output_file)
        print('Performance: {:.2f} MIt/s'.format(self.results['Performance [MIt/s]']),
              file=output_file)
        if self.verbose > 0:
            print('MEM bandwidth: {:.2f} MByte/s'.format(self.results['MEM BW [MByte/s]']),
                  file=output_file)
        print('', file=output_file)

        if not self.no_phenoecm:
            print("Data Transfers:")
            print("{:^8} |".format("cache"), end='')
            for metrics in self.results['data transfers'].values():
                for metric_name in sorted(metrics):
                    print(" {:^14}".format(metric_name), end='')
                print()
                break
            for cache, metrics in sorted(self.results['data transfers'].items()):
                print("{!s:^8} |".format(cache), end='')
                for k, v in sorted(metrics.items()):
                    print(" {!s:^14}".format(v), end='')
                print()
            print()

            print('Phenomenological ECM model: {{ {T_OL:.1f} || {T_nOL:.1f} | {T_L1L2:.1f} | '
                  '{T_L2L3:.1f} | {T_L3MEM:.1f} }} cy/CL'.format(
                **{k: float(v) for k, v in self.results['ECM'].items()}),
                file=output_file)
            print('T_OL assumes that two loads per cycle may be retired, which is true for '
                  '128bit SSE/half-AVX loads on SNB and IVY, and 256bit full-AVX loads on HSW, '
                  'BDW, SKL and SKX, but it also depends on AGU availability.',
                  file=output_file)
