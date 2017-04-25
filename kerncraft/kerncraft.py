#!/usr/bin/env python

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

# Version check
import sys
if sys.version_info[0] == 2 and sys.version_info < (2, 7) or \
        sys.version_info[0] == 3 and sys.version_info < (3, 4):
    print("Must use python 2.7 or 3.4 and greater.", file=sys.stderr)
    sys.exit(1)

import argparse
import os.path
import pickle
import shutil
import math
import re
import itertools
import operator
from functools import reduce

from .pycparser import clean_code
import sympy
import six
from six.moves import range
from ruamel import yaml

from . import models
from .kernel import KernelCode, KernelDescription
from .machinemodel import MachineModel


def space(start, stop, num, endpoint=True, log=False, base=10):
    '''
    Returns list of evenly spaced integers over an interval.

    Numbers can either be evenlty distributed in a linear space (if *log* is False) or in a log
    space (if *log* is True). If *log* is True, base is used to define the log space basis.

    If *endpoint* is True, *stop* will be the last retruned value, as long as *num* >= 2.
    '''
    assert type(start) is int and type(stop) is int and type(num) is int, \
        "start, stop and num need to be intergers"
    assert num >= 2, "num has to be atleast 2"

    if log:
        start = math.log(start, base)
        stop = math.log(stop, base)

    if endpoint:
        steplength = float((stop-start))/float(num-1)
    else:
        steplength = float((stop-start))/float(num)

    i = 0
    while i < num:
        if log:
            yield int(round(base**(start + i*steplength)))
        else:
            yield int(round(start + i*steplength))
        i += 1


class AppendStringRange(argparse.Action):
    """
    Action to append a string and a range discription

    A range discription must have the following format: start[-stop[:num[log[base]]]]
    if stop is given, a list of integers is compiled
    if num is given, an evently spaced lsit of intergers from start to stop is compiled
    if log is given, the integers are evenly spaced on a log space
    if base is given, the integers are evently spaced on that base (default: 10)
    """
    def __call__(self, parser, namespace, values, option_string=None):
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
                    values[1] = list(range(int(gd['start']), int(gd['stop'])+1))
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
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--asm-increment', metavar='INCR', default=0, type=int,
                        help='Increment of stor pointer within one ASM block in bytes. If 0, '
                             'automatic detetection will be used and can lead to user input being '
                             'required.')
    parser.add_argument('--store', metavar='PICKLE', type=argparse.FileType('a+b'),
                        help='Addes results to PICKLE file for later processing.')
    parser.add_argument('--unit', '-u', choices=['cy/CL', 'cy/It', 'It/s', 'FLOP/s'],
                        help='Select the output unit, defaults to model specific if not given.')
    parser.add_argument('--cores', '-c', metavar='CORES', type=int, default=1,
                        help='Number of cores to be used in parallel. (default: 1)')
    parser.add_argument('--kernel-description', action='store_true',
                        help='Use kernel description instead of analyzing the kernel code.')

    # Needed for ECM, ECMData and Roofline model:
    parser.add_argument('--cache-predictor', '-P', choices=['LC', 'SIM'], default='SIM',
                        help='Change cache predictor to use, options are LC (layer conditions) and '
                             'SIM (cache simulation with pycachesim), default is SIM.')

    for m in models.__all__:
        ag = parser.add_argument_group('arguments for '+m+' model', getattr(models, m).name)
        getattr(models, m).configure_arggroup(ag)

    return parser


def check_arguments(args, parser):
    if args.asm_block not in ['auto', 'manual']:
        try:
            args.asm_block = int(args.asm_block)
        except ValueError:
            parser.error('--asm-block can only be "auto", "manual" or an integer')


def run(parser, args, output_file=sys.stdout):
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
    machine = MachineModel(args.machine.name)

    # process kernel
    if not args.kernel_description:
        code = six.text_type(args.code_file.read())
        code = clean_code(code)
        kernel = KernelCode(code, filename=args.code_file.name)
    else:
        description = six.text_type(args.code_file.read())
        kernel = KernelDescription(yaml.load(description))

    # if no defines were given, guess suitable defines in-mem
    # TODO support in-cache
    # TODO broaden cases to n-dimensions
    # TODO make configurable (no hardcoded 512MB/1GB/min. 3 iteration ...)
    # works only for up to 3 dimensions
    if not args.define:
        required_consts = [v[1] for v in kernel.variables.values() if v[1] is not None]
        assert all([1 <= len(rc) <= 3 for rc in required_consts]), "Automatic selection of " + \
            "defines only works with up to 3 dimensions."
        inner_loop_syms = kernel._loop_stack[-1][2].free_symbols
        assert len(inner_loop_syms) == 1, "Automatic selection can only work, if " + \
            "inner-most loop's max statement contains exactly one constant/define (e.g. N)."
        inner_loop_const = inner_loop_syms.pop()
        define_product = []

        # From 100 elements to 512MB of data with 150 data points on log10 scale
        for inner_dim_size in space(
                100, int(0.5*1025**3/kernel.datatypes_size[kernel.datatype]), 150, log=True):
            # we choose all other constants, such that the largest array consumes 1-3GB of memory:
            array_dims = sorted(required_consts, key=len)[-1]
            array_size = reduce(operator.mul, array_dims).subs(inner_loop_const, inner_dim_size)
            assert 0 <= len(array_size.free_symbols) <= 1, "Automatic selection can only  " + \
                "work, if arrays depend only on the inner-loop constant and one more " + \
                "constant (at most)."

            if len(array_size.free_symbols) == 1:
                define_product.append([
                    (inner_loop_const, inner_dim_size),
                    (array_size.free_symbols.pop(),
                     max(int(sympy.solve(sympy.Eq(array_size, 1024**3))[0]), 3))]) # min 3 it.
            else:
                define_product.append([(inner_loop_const, inner_dim_size)])
    else:
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

    for define in define_product:
        # Reset state of kernel
        kernel.clear_state()

        # Add constants from define arguments
        for k, v in define:
            kernel.set_constant(k, v)

        for model_name in set(args.pmodel):
            # print header
            print('{:=^80}'.format(' kerncraft '), file=output_file)
            print('{:<40}{:>40}'.format(args.code_file.name, '-m '+args.machine.name),
                  file=output_file)
            print(' '.join(['-D {} {}'.format(k,v) for k,v in define]), file=output_file)
            print('{:-^80}'.format(' '+model_name+' '), file=output_file)

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
            tempname = args.store.name + '.tmp'
            with open(tempname, 'wb+') as f:
                pickle.dump(result_storage, f)
            shutil.move(tempname, args.store.name)


def main():
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
