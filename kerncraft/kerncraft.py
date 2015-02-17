#!/usr/bin/env python

from __future__ import print_function

import argparse
import ast
import sys
import os.path

import models
from kernel import Kernel
from machinemodel import MachineModel


class AppendStringInteger(argparse.Action):
    """Action to append a string and integer"""
    def __call__(self, parser, namespace, values, option_string=None):
        message = ''
        if len(values) != 2:
            message = 'requires 2 arguments'
        else:
            try:
                values[1] = int(values[1])
            except ValueError:
                message = ('second argument requires an integer')

        if message:
            raise argparse.ArgumentError(self, message)

        if hasattr(namespace, self.dest):
            getattr(namespace, self.dest).append(values)
        else:
            setattr(namespace, self.dest, [values])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--machine', '-m', type=file, required=True,
                        help='Path to machine description yaml file.')
    parser.add_argument('--pmodel', '-p', choices=models.__all__, required=True, action='append',
                        default=[], help='Performance model to apply')
    parser.add_argument('-D', '--define', nargs=2, metavar=('KEY', 'VALUE'), default=[],
                        action=AppendStringInteger,
                        help='Define constant to be used in C code. Values must be integers. '
                             'Overwrites constants from testcase file.')
    parser.add_argument('--testcases', '-t', action='store_true',
                        help='Use testcases file.')
    parser.add_argument('--testcase-index', '-i', metavar='INDEX', type=int, default=0,
                        help='Index of testcase in testcase file. If not given, all cases are '
                             'executed.')
    parser.add_argument('--verbose', '-v', action='count',
                        help='Increases verbosity level.')
    parser.add_argument('code_file', metavar='FILE', type=argparse.FileType(), nargs='+',
                        help='File with loop kernel C code')
    parser.add_argument('--asm-block', metavar='BLOCK', default='auto',
                        help='Number of ASM block to mark for IACA, "auto" for automatic '
                             'selection or "manual" for interactiv selection.')
    for m in models.__all__:
        ag = parser.add_argument_group('arguments for '+m+' model', getattr(models, m).name)
        getattr(models, m).configure_arggroup(ag)

    # BUSINESS LOGIC IS FOLLOWING
    args = parser.parse_args()
    
    # Checking arguments
    if args.asm_block not in ['auto', 'manual']:
        try:
            args.asm_block = int(args.asm_block)
        except ValueError:
            parser.error('--asm-block can only be "auto", "manual" or an integer')

    # machine information
    # Read machine description
    machine = MachineModel(args.machine.name)

    # process kernels and testcases
    for code_file in args.code_file:
        code = code_file.read()

        # Add constants from testcase file
        if args.testcases:
            testcases_file = open(os.path.splitext(code_file.name)[0]+'.testcases')
            testcases = ast.literal_eval(testcases_file.read())
            if args.testcase_index:
                testcases = [testcases[args.testcase_index]]
        else:
            testcases = [{'constants': {}}]

        for testcase in testcases:

            print('='*80 + '\n{:^80}\n'.format(code_file.name) + '='*80)

            kernel = Kernel(code, filename=code_file.name)

            assert 'constants' in testcase, "Could not find key 'constants' in testcase file."
            for k, v in testcase['constants']:
                kernel.set_constant(k, v)

            # Add constants from define arguments
            for k, v in args.define:
                kernel.set_constant(k, v)

            kernel.process()
            kernel.print_kernel_code()
            print()
            kernel.print_variables_info()
            kernel.print_constants_info()
            kernel.print_kernel_info()

            for model_name in set(args.pmodel):
                model = getattr(models, model_name)(kernel, machine, args, parser)

                model.analyze()
                model.report()

                # TODO take care of different performance models
                if 'results-to-compare' in testcase:
                    failed = False
                    for key, value in model.results.items():
                        if key in testcase['results-to-compare']:
                            correct_value = float(testcase['results-to-compare'][key])
                            diff = abs(value - correct_value)
                            if diff > correct_value*0.1:
                                print("Test values did not match: {} should have been {}, but was "
                                      "{}.".format(key, correct_value, value))
                                failed = True
                            elif diff:
                                print("Small difference from theoretical value: {} should have "
                                      "been {}, but was {}.".format(key, correct_value, value))
                    if failed:
                        sys.exit(1)

if __name__ == '__main__':
    main()
