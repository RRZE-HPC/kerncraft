#!/usr/bin/env python3
"""Comand line interface of Kerncraft."""
# Version check
import sys
import argparse
import os.path
import pickle
import shutil
import math
import re
import itertools
from datetime import datetime
from functools import lru_cache

from ruamel import yaml

from . import models
from . import __version__
from .kernel import KernelCode, KernelDescription, symbol_pos_int
from .machinemodel import MachineModel
from .pycparser_utils import clean_code


def space(start, stop, num, endpoint=True, log=False, base=10):
    """
    Return list of evenly spaced integers over an interval.

    Numbers can either be evenly distributed in a linear space (if *log* is False) or in a log
    space (if *log* is True). If *log* is True, base is used to define the log space basis.

    If *endpoint* is True, *stop* will be the last retruned value, as long as *num* >= 2.
    """
    assert type(start) is int and type(stop) is int and type(num) is int, \
        "start, stop and num need to be intergers"
    assert num >= 2, "num has to be atleast 2"

    if log:
        start = math.log(start, base)
        stop = math.log(stop, base)

    if endpoint:
        step_length = float((stop - start)) / float(num - 1)
    else:
        step_length = float((stop - start)) / float(num)

    i = 0
    while i < num:
        if log:
            yield int(round(base ** (start + i * step_length)))
        else:
            yield int(round(start + i * step_length))
        i += 1


def int_or_str(s):
    """Casts string to int if possible, otherwise return original string."""
    try:
        return int(s)
    except ValueError:
        return s


def uniquify(l):
    # Uniquify list while preserving order
    seen = set()
    return [x for x in l if x not in seen and not seen.add(x)]


@lru_cache()
def get_last_modified_datetime(dir_path=os.path.dirname(__file__)):
    """Return datetime object of latest change in kerncraft module directory."""
    max_mtime = 0
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            p = os.path.join(root, f)
            try:
                max_mtime = max(max_mtime, os.stat(p).st_mtime)
            except FileNotFoundError:
                pass
    return datetime.utcfromtimestamp(max_mtime)


class AppendStringRange(argparse.Action):
    """
    Argparse Action to append integer range from string.

    A range description must have the following format: start[-stop[:num[log[base]]]]
    if stop is given, a list of integers is compiled
    if num is given, an evenly spaced list of integers from start to stop is compiled
    if log is given, the integers are evenly spaced on a log space
    if base is given, the integers are evenly spaced on that base (default: 10)
    """

    def __call__(self, parser, namespace, values, option_string=None):
        """Execute action."""
        message = ''
        if len(values) != 2:
            message = 'requires 2 arguments'
        else:
            m = re.match(r'(?P<start>\d+)(?:-(?P<stop>\d+)(?::(?P<num>\d+)'
                         r'(:?(?P<log>log)(:?(?P<base>\d+))?)?)?)?',
                         values[1])
            if m:
                gd = m.groupdict()
                if gd['stop'] is None:
                    values[1] = [int(gd['start'])]
                elif gd['num'] is None:
                    values[1] = list(range(int(gd['start']), int(gd['stop']) + 1))
                else:
                    log = gd['log'] is not None
                    base = int(gd['base']) if gd['base'] is not None else 10
                    values[1] = space(
                        int(gd['start']), int(gd['stop']), int(gd['num']), log=log, base=base)
            else:
                message = 'second argument must match: start[-stop[:num[log[base]]]]'

        if message:
            raise argparse.ArgumentError(self, message)

        if hasattr(namespace, self.dest):
            getattr(namespace, self.dest).append(values)
        else:
            setattr(namespace, self.dest, [values])


def create_parser():
    """Return argparse parser."""
    parser = argparse.ArgumentParser(
        description='Analytical performance modelling and benchmarking toolkit.',
        epilog='For help, examples, documentation and bug reports go to:\nhttps://github.com'
               '/RRZE-HPC/kerncraft\nLicense: AGPLv3')
    parser.add_argument('--version', action='version', version='%(prog)s {}'.format(__version__))
    parser.add_argument('--machine', '-m', type=argparse.FileType('r'), required=True,
                        help='Path to machine description yaml file.')
    parser.add_argument('--pmodel', '-p', choices=models.__all__, required=True, action='append',
                        default=[], help='Performance model to apply')
    parser.add_argument('-D', '--define', nargs=2, metavar=('KEY', 'VALUE'), default=[],
                        action=AppendStringRange,
                        help='Define constant to be used in C code. Values must be integer or '
                             'match start-stop[:num[log[base]]]. If range is given, all '
                             'permutation s will be tested. Overwrites constants from testcase '
                             'file.')
    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help='Increases verbosity level.')
    parser.add_argument('code_file', metavar='FILE', type=argparse.FileType(),
                        help='File with loop kernel C code')
    parser.add_argument('--asm-block', metavar='BLOCK', default='auto',
                        help='Number of ASM block to mark for IACA, "auto" for automatic '
                             'selection or "manual" for interactiv selection.')
    parser.add_argument('--pointer-increment', metavar='INCR', default='auto', type=int_or_str,
                        help='Increment of store pointer within one ASM block in bytes. If "auto": '
                             'automatic detection, error on failure to detect, if '
                             '"auto_with_manual_fallback": fallback to manual input, or if '
                             '"manual": always prompt user.')
    parser.add_argument('--store', metavar='PICKLE', type=argparse.FileType('a+b'),
                        help='Addes results to PICKLE file for later processing.')
    parser.add_argument('--unit', '-u', choices=['cy/CL', 'cy/It', 'It/s', 'FLOP/s'],
                        help='Select the output unit, defaults to model specific if not given.')
    parser.add_argument('--cores', '-c', metavar='CORES', type=int, default=1,
                        help='Number of cores to be used in parallel. (default: 1) '
                             'ECM model will consider the scaling of the last level cache and '
                             'predict the overall performance in addition to single-core behavior. '
                             'The benchmark mode will run the code with OpenMP on as many physical '
                             'cores.')
    parser.add_argument('--kernel-description', action='store_true',
                        help='Use kernel description instead of analyzing the kernel code.')
    parser.add_argument('--clean-intermediates', action='store_true',
                        help='If set, will delete all intermediate files after completion.')

    # Needed for ECM, ECMData and Roofline model:
    parser.add_argument('--cache-predictor', '-P', choices=['LC', 'SIM'], default='SIM',
                        help='Change cache predictor to use, options are LC (layer conditions) and '
                             'SIM (cache simulation with pycachesim), default is SIM.')

    # Needed for ECM, RooflineIACA and Benchmark model:
    parser.add_argument('--compiler', '-C', type=str, default=None,
                        help='Compiler to use, default is first in machine description file.')
    parser.add_argument('--compiler-flags', type=str, default=None,
                        help='Compiler flags to use. If not set, flags are taken from machine '
                             'description file (-std=c99 is always added).')

    for m in models.__all__:
        ag = parser.add_argument_group('arguments for ' + m + ' model', getattr(models, m).name)
        getattr(models, m).configure_arggroup(ag)

    return parser


def check_arguments(args, parser):
    """Check arguments passed by user that are not checked by argparse itself."""
    if args.asm_block not in ['auto', 'manual']:
        try:
            args.asm_block = int(args.asm_block)
        except ValueError:
            parser.error('--asm-block can only be "auto", "manual" or an integer')

    # Set default unit depending on performance model requested
    if not args.unit:
        if 'Roofline' in args.pmodel or 'RooflineIACA' in args.pmodel:
            args.unit = 'FLOP/s'
        else:
            args.unit = 'cy/CL'


def run(parser, args, output_file=sys.stdout):
    """Run command line interface."""
    # Try loading results file (if requested)
    result_storage = {}
    if args.store:
        args.store.seek(0)
        try:
            result_storage = pickle.load(args.store)
        except EOFError:
            pass
        args.store.close()

    # machine information
    # Read machine description
    machine = MachineModel(args.machine.name, args=args)

    # process kernel
    if not args.kernel_description:
        code = str(args.code_file.read())
        code = clean_code(code)
        kernel = KernelCode(code, filename=args.code_file.name, machine=machine,
                            keep_intermediates=not args.clean_intermediates)
    else:
        description = str(args.code_file.read())
        kernel = KernelDescription(yaml.load(description, Loader=yaml.Loader), machine=machine)

    # define constants
    required_consts = [v[1] for v in kernel.variables.values() if v[1] is not None]
    required_consts += [[l['start'], l['stop']] for l in kernel.get_loop_stack()]
    # split into individual consts
    required_consts = [i for l in required_consts for i in l]
    required_consts = set([i for l in required_consts for i in l.free_symbols])
    if len(required_consts) > 0:
        # build defines permutations
        define_dict = {}
        for name, values in args.define:
            if name not in define_dict:
                define_dict[name] = [[name, v] for v in values]
                continue
            for v in values:
                if v not in define_dict[name]:
                    define_dict[name].append([name, v])
        define_product = list(itertools.product(*list(define_dict.values())))

        # Check that all consts have been defined
        if set(required_consts).difference(set([symbol_pos_int(k) for k in define_dict.keys()])):
            raise ValueError("Not all constants have been defined. Required are: {}".format(
                required_consts))
    else:
        define_product = [{}]

    for define in define_product:
        # Reset state of kernel
        kernel.clear_state()

        # Add constants from define arguments
        for k, v in define:
            kernel.set_constant(k, v)

        for model_name in uniquify(args.pmodel):
            # print header
            print('{:^80}'.format(' kerncraft '), file=output_file)
            print('{:<40}{:>40}'.format(args.code_file.name, '-m ' + args.machine.name),
                  file=output_file)
            print(' '.join(['-D {} {}'.format(k, v) for k, v in define]), file=output_file)
            print('{:-^80}'.format(' ' + model_name + ' '), file=output_file)

            if args.verbose > 1:
                if not args.kernel_description:
                    kernel.print_kernel_code(output_file=output_file)
                    print('', file=output_file)
                kernel.print_variables_info(output_file=output_file)
                kernel.print_kernel_info(output_file=output_file)
            if args.verbose > 0:
                kernel.print_constants_info(output_file=output_file)

            model = getattr(models, model_name)(kernel, machine, args, parser)

            model.analyze()
            model.report(output_file=output_file)

            # Add results to storage
            kernel_name = os.path.split(args.code_file.name)[1]
            if kernel_name not in result_storage:
                result_storage[kernel_name] = {}
            if tuple(kernel.constants.items()) not in result_storage[kernel_name]:
                result_storage[kernel_name][tuple(kernel.constants.items())] = {}
            result_storage[kernel_name][tuple(kernel.constants.items())][model_name] = \
                model.results

            print('', file=output_file)

        # Save storage to file (if requested)
        if args.store:
            temp_name = args.store.name + '.tmp'
            with open(temp_name, 'wb+') as f:
                pickle.dump(result_storage, f)
            shutil.move(temp_name, args.store.name)


def main():
    """Initialize and run command line interface."""
    # Create and populate parser
    parser = create_parser()

    # Parse given arguments
    args = parser.parse_args()

    # Checking arguments
    check_arguments(args, parser)

    # BUSINESS LOGIC IS FOLLOWING
    run(parser, args)


if __name__ == '__main__':
    main()
