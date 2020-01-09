#!/usr/bin/env python3
"""Stand-alone version of kerncraft's benchmark model."""
import sys
import argparse
import os.path
import pickle
import shutil
import math
import re
import itertools

from collections import OrderedDict
from ruamel import yaml

from . import __version__
from .kerncraft import space

from .models.standalone_benchmark import StandaloneBenchmark
from .kernel import BinaryDescription, symbol_pos_int
from .machinemodel import MachineModel


class AppendStringRange(argparse.Action):
    """
    Argparse Action to append integer range from string.

    A range description must have the following format: [[...,]region:]start[:end[:scaling]]
    'region' defaults to the empty string
    if 'end' is omitted, the variable is assumed fixed
    if not given, 'scaling' is assumed based on the choice of start and end value, i.e.
                    linearly increasing, if start < end
                    logarithmically decreasing, if start > end
    """

    def __call__(self, parser, namespace, values, option_string=None):
        """Execute action."""
        message = ''
        variables = {}

        if len(values) != 2:
            message = 'requires 2 arguments'
        else:
            # optional: likwid region(s) for which this variable is specified
            m = re.match(r'(?:(?P<region>(\D\w*,)*\D\w*):)?' 
                         r'(?:(?P<start_var>[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)?:?)'
                         r'(?:(?P<stop_var>[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)?\:?)?'
                         r'(?:(?P<scale_var>log|lin)?)?', values[1])

            if m:
                gd = m.groupdict()

                if gd['start_var'] is not None:

                    # get likwid region(s)
                    regions = gd['region'].split(',') if gd['region'] is not None else ['']
                    variables['region'] = regions

                    start = float(gd['start_var'])
                    variables['start'] = start
                    variables['value'] = start

                    if gd['stop_var'] is None:
                        variables['adjustable'] = False
                    else:
                        variables['adjustable'] = True
                        stop = float(gd['stop_var'])
                        variables['stop'] = stop

                        if gd['scale_var'] is None:
                            variables['scale'] = 'lin' if stop > start else 'log'
                            print("WARNING: You did not specify a scaling for your adjustable "
                                  "variable. From the choice of your start and stop values, {} "
                                  "scaling is assumed.".format(variables['scale']))
                        else:
                            variables['scale'] = gd['scale_var']
            else:
                message = 'second argument must match: [[...,]region:]start[:stop[:scale]]'

        if message:
            raise argparse.ArgumentError(self, message)

        if hasattr(namespace, self.dest):
            # getattr(namespace, self.dest).append(values)
            getattr(namespace, self.dest)[values[0]] = variables
        else:
            # setattr(namespace, self.dest, [values])
            setattr(namespace, self.dest, OrderedDict(values[0], variables))


class AppendLoopRange(argparse.Action):
    """
    Argparse Action to append loop range from string.

    A range description must have the following format: '[[...,]region:]start:[step:]end[:variable]'
    'region' defaults to the empty string
    'step' defaults to 1
    'variable' defaults to None
    """

    def __call__(self, parser, namespace, values, option_string=None):
        """Execute action."""

        message = ''

        # optional: likwid region(s) for which this loop range is specified
        m = re.match(r'(?:(?P<region>(\D\w*,)*\D\w*):)?'
                     # lower bound of loop
                     r'(?P<start>\d+\.?\d*):'
                     # optional: step size
                     r'((?P<step>\d+\.?\d*):)?'
                     # upper bound of loop
                     r'(?P<end>\d+\.?\d*)'
                     # optional: variable that is responsible for the loop length this is needed,
                     # when an adjustable variable controls the upper bound of a loop
                     r'(:(?P<variable>\D\S*))?', 
                     values, flags=re.VERBOSE)
        if m:
            gd = m.groupdict()

            # get (optional) region(s), defaults to empty string
            regions = gd['region'].split(',') if gd['region'] is not None else ['']

            if gd['step'] is None:
                gd['step'] = 1
            try:
                start = int(gd['start'])
                end   = int(gd['end'])
                step  = int(gd['step'])

                values = {'start': start, 'end': end, 'step': step, 'variable': gd['variable'], 
                          'offset': None}
            except ValueError:
                print("Pattern of loop range must match "
                      "'[[...,]region:]start:[step:]end[:variable]'")
        else:
            message = "argument must match: 'start:[step:]end|marker'"
            raise argparse.ArgumentError(self, message)

        if hasattr(namespace, self.dest):
            # another loop has already been defined
            for region in regions:
                if region in getattr(namespace, self.dest).keys():
                    # another loop has already been defined for this region
                    getattr(namespace, self.dest)[region].append(values)
                else:
                    # first loop to be defined for this region
                    getattr(namespace, self.dest)[region] = [values]
        else:
            # first loop to be defined
            setattr(namespace, self.dest, [values])


class AppendRepetitionDefines(argparse.Action):
    """
    Argparse Action to append loop range from string.

    A range description must have the following format: '[region:]qualifier'
    If no region is specified, it is assumed that the qualifier is the number of repetitions for all regions.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        """Execute action."""

        message = ''
        regions = []
        qualifier = None
        if len(values) == 0:
            message = 'requires 1 argument'
        else:
            # optionally: likwid region(s) for which the number of repetitions is defined
            m = re.match(r'(?:(?P<region>(\D\w*,)*\D\w*):)?'    
                         # qualifier for the repetitions. Can either be a fixed number, a variable 
                         # name or the specifier 'marker'
                         r'(?P<qualifier>\w+)$',                
                         values)
            if m:
                gd = m.groupdict()
                try:
                    regions = gd['region'].split(',') if gd['region'] is not None else ['']
                    qualifier = gd['qualifier']

                except ValueError:
                    print("Pattern of loop range must match '[[...,]region:]qualifier'")
            else:
                message = "argument must match: '[[...,]region:]qualifier'"

        if message:
            raise argparse.ArgumentError(self, message)

        if hasattr(namespace, self.dest):
            for region in regions:
                getattr(namespace, self.dest)[region] = qualifier

        else:
            setattr(namespace, self.dest, {regions[0]: qualifier})
            for region in regions[1:]:
                getattr(namespace, self.dest)[region] = qualifier


class AppendFlops(argparse.Action):
    """
    Argparse Action to append flops for optionally a specific likwid region.

    A flop description mist have the following format: '[[...,]region:]flops'.
    If no region is defined, it is assumed that the number of flops holds for all regions (?!)
    """
    def __call__(self, parser, namespace, values, option_string=None):
        """Execute action."""

        message = ''
        regions = []
        flops = 0

        if len(values) == 0:
            message = 'requires 1 argument'
        else:
            m = re.match(r'(?:(?P<region>(\D\w*,)*\D\w*):)?(?P<flops>\d+)$',
                         values)
            if m:
                gd = m.groupdict()
                if gd['region'] is None:
                    gd['region'] = ''
                try:
                    regions = gd['region'].split(',')
                    flops = int(gd['flops'])

                except ValueError:
                    print("Pattern of flops must match '[[...,]region:]flops'")
            else:
                message = "argument must match: '[[...,]region:]flops'"

        if message:
            raise argparse.ArgumentError(self, message)

        if hasattr(namespace, self.dest):
            for region in regions:
                if region in getattr(namespace, self.dest).keys():
                    raise Exception('FLOPs can only assigned once per region.')
                else:
                    getattr(namespace, self.dest)[region] = flops
        else:
            setattr(namespace, self.dest, {regions[0]: flops})
            for region in regions[1:]:
                getattr(namespace, self.dest)[region] = flops


class ParseMarkers(argparse.Action):
    """
    Splits the given likwid marker if any given.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        """Execute action."""

        marker = {'use_marker': True, 'region': []}

        if values:
            # region names are specified
            marker['region'] = (values.split(','))

        setattr(namespace, self.dest, marker)


def create_parser():
    """Return argparse parser."""
    parser = argparse.ArgumentParser(
        description="Kerncraft's stand-alone benchmarking tool.",
        epilog='For help, examples, documentation, and bug reports go to '
                  'https://github.com/RRZE-HPC/kerncraft\nLicense: AGPLv3')

    parser.add_argument('--version', action='version', version='%(prog)s {}'.format(__version__))
    parser.add_argument('--machine', '-m', type=argparse.FileType('r'), required=True,
                        help='Path to machine description yaml file.')
    parser.add_argument('--define', '-D', nargs=2, metavar=('KEY', 'VALUE'), default=OrderedDict(),
                        action=AppendStringRange,
                        help='Define constant to be used in C code. Values must be integer or '
                             'match start-stop[:num[log[base]]]. If range is given, all '
                             'permutations will be tested. Overwrites constants from testcase '
                             'file.\nConstants with leading underscores are adaptable, i.e., '
                             'they can be change to enforce a minimum runtime.')
    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help='Increase verbosity level.')
    parser.add_argument('--store', metavar='PICKLE', type=argparse.FileType('a+b'),
                        help='Adds results to PICKLE file for later processing.')
    parser.add_argument('--unit', '-u', choices=['cy/CL', 'cy/It', 'It/s', 'FLOP/s'],
                        help='Select the output unit, defaults to model specific if not given.')
    parser.add_argument('--cores', '-c', metavar='CORES', type=int, default=1,
                        help='Number of cores to be used in parallel. (default: 1) The benchmark '
                             'model will run the code with OpenMP on as many physical cores.')
    parser.add_argument('binary', metavar='FILE',
                        help='Binary to be benchmarked.')
    parser.add_argument('--clean-intermediates', action='store_true',
                        help='If set will delete all intermediate files after completion.')
    parser.add_argument('--compiler', '-C', type=str, default=None,
                        help='Compiler to use, default is first in machine description file.')
    parser.add_argument('--compiler-flags', type=str, default=None,
                        help='Compiler flags to use. If not set, flags are taken from machine '
                            'description file (-std=c99 is always added).')
    parser.add_argument('--datatype', metavar='DATATYPE', type=str, choices=['float', 'double'], 
                        default='double',
                        help="Datatype of sources and destinations of the kernel. Defaults to "
                             "'double'.")
    parser.add_argument('--flops', metavar='FLOPS', required=True, action=AppendFlops, default={},
                        help='Number of floating-point operations per inner-most iteration of the '
                             'kernel.')
    parser.add_argument('--loop', '-L', metavar='LOOP_RANGE', required=True, action=AppendLoopRange,
                        default={},
                        help="Define ranges of nested loops. The definition must match "
                             "'start:[step:]end'. 'step' defaults to 1.")
    parser.add_argument('--repetitions', '-R', metavar='REPETITIONS',
                        action=AppendRepetitionDefines, default={'': '1'},
                        help='Number of kernel repetitions. Can be either a fixed number, a '
                             "variable name, or the specifier 'marker'. Specifying a variable "
                             "name, the number of repetitions is automatically adjusted when the "
                             "variable is changed. Specifying 'marker', the number of repetitions "
                             "is obtained from likwid-perfctr.")
    parser.add_argument('--marker', action=ParseMarkers, nargs='?',
                        default={'use_marker': False, 'region': ['']},
                        help='Benchmark using likwid markers.')

    ag = parser.add_argument_group('arguments for stand-alone benchmark model', 'benchmark')
    StandaloneBenchmark.configure_arggroup(ag)

    return parser

def check_arguments(args, parser):
    """Check arguments passed by user that are not checked by argparse itself."""

    # set default unit
    if not args.unit:
        args.unit = 'cy/CL'

def run(parser, args, output_file = sys.stdout):
    """Run command line interface."""
    # try loading  results file (if requested)
    result_storage = {}
    if args.store:
        args.store.seek(0)
        try:
            result_storage = pickle.load(args.store)
        except EOFError:
            pass
        args.store.close()

    # read machine description
    machine  = MachineModel(args.machine.name, args=args)

    # process kernel description
    kernel = BinaryDescription(args=args, machine=machine)

    # for define in define_product:
    # reset state of kernel
    kernel.clear_state()

    # add constants from define arguments

    for k, v in args.define.items():
        kernel.set_constant(k, v['value'])

    # set up benchmark
    model = StandaloneBenchmark(kernel, machine, args, parser)

    model.analyze()

    # print header
    print('\n\n{:^80}'.format(' stand-alone kerncraft benchmark '), file=output_file)
    print('{:<40}{:>40}'.format(kernel.binary, ' -m ' + args.machine.name),
          file=output_file)
    print(' '.join(['-D {} {}'.format(k, v['value']) for k, v in args.define.items()]), 
          file=output_file)
    print('{:-^80}'.format(' ' + args.binary + ' '), file=output_file)

    if args.verbose > 1:
        kernel.print_kernel_info(output_file = output_file)

    if args.verbose > 0:
        kernel.print_constants_info(output_file = output_file)

    model.report(output_file = output_file)

    # add results to storage
    bin_name = os.path.split(kernel.binary)[1]
    if bin_name not in result_storage:
        result_storage[bin_name] = {}
    if tuple(kernel.constants.items()) not in result_storage[bin_name]:
        result_storage[bin_name][tuple(kernel.constants.items())] = {}
    result_storage[bin_name][tuple(kernel.constants.items())][model.name] = model.results

    print('', file = output_file)

    # save storage to file (if requested)
    if args.store:
        temp_name = args.store.name + '.tmp'
        with open(temp_name, 'wb+') as f:
            pickle.dump(result_storage, f)
        shutil.move(temp_name, args.store.name)


def main():
    """Initialize and run command line interface."""
    # create and populate parser
    parser = create_parser()

    # parse given arguments
    args = parser.parse_args()

    # checking arguments
    check_arguments(args, parser)

    # run benchmarking
    run(parser, args)


if __name__ == '__main__':
    main()
