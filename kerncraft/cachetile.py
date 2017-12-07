#!/usr/bin/env python3
import argparse
import sys

import sympy
from ruamel import yaml

from . import models
from .kernel import KernelDescription
from .machinemodel import MachineModel


def create_parser():
    parser = argparse.ArgumentParser(description='Find optimal tiling sizes using the ECMData '
                                                 'model.')
    parser.add_argument('--machine', '-m', type=argparse.FileType('r'), required=True,
                        help='Path to machine description yaml file.')
    parser.add_argument('--define', '-D', nargs=2, metavar=('KEY', 'VALUE'), default=[],
                        action='append',
                        help='Define fixed constants. Values must be integer.')
    parser.add_argument('--min-block-length', '-b', type=int, metavar='MIN', default=100)
    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help='Increases verbosity level.')
    parser.add_argument('--cores', '-c', metavar='CORES', type=int, default=1,
                        help='Number of cores to be used in parallel. (default: 1)')
    parser.add_argument('description_file', metavar='FILE', type=argparse.FileType(),
                        help='File with loop kernel description in YAML')
    return parser


def simulate(kernel, model, define_dict, blocking_constant, blocking_length):
    """Setup and execute model with given blocking length"""
    kernel.clear_state()

    # Add constants from define arguments
    for k, v in define_dict.items():
        kernel.set_constant(k, v)

    kernel.set_constant(blocking_constant, blocking_length)

    model.analyze()
    return sum([cy for dscr, cy in model.results['cycles']])


def run(parser, args):
    # machine information
    # Read machine description
    machine = MachineModel(args.machine.name)

    # process kernel description
    description = str(args.description_file.read())
    kernel = KernelDescription(yaml.load(description))

    # Add constants from define arguments
    define_dict = {}
    for name, value in args.define:
        assert name not in define_dict, "Redefinition of constants is not allowed."
        define_dict[name] = int(value)

    model = models.ECMData(kernel, machine, args, parser)

    # Select constant to search blocksize for
    undefined_constants = set()
    for var_name, var_info in kernel.variables.items():
        var_type, var_size = var_info
        for size in var_size:
            for s in size.atoms(sympy.Symbol):
                if s.name not in define_dict:
                    undefined_constants.add(s)
    assert len(undefined_constants) == 1, "There are multiple or none undefined constants {!r}. " \
        "Exactly one must be undefined.".format(undefined_constants)
    blocking_constant = undefined_constants.pop()

    if args.verbose >= 1:
        print("blocking constant:", blocking_constant)

    # min and max block lengths
    min_length = args.min_block_length
    min_runtime = simulate(kernel, model, define_dict, blocking_constant, min_length)

    # determain max search length
    # upper bound: number of floats that fit into the last level cache
    max_length = int(machine['memory hierarchy'][-2]['size per group'])//4
    if args.verbose >= 1:
        print("upper search bound:", max_length)
    length = min_length*3
    while length < max_length:
        runtime = simulate(kernel, model, define_dict, blocking_constant, length)
        if args.verbose >= 1:
            print("min", min_length, min_runtime, "current", length, runtime, "max", max_length)

        # Increase search window
        if runtime > min_runtime:
            max_length = length  # and break
        else:
            length *= 2  # continue search

    # search end of block
    while max_length - min_length > 10:
        # Take median for benchmark:
        length = (max_length - min_length) // 2 + min_length

        # Execute simulation
        runtime = simulate(kernel, model, define_dict, blocking_constant, length)
        if args.verbose >= 1:
            print("min", min_length, min_runtime, "current", length, runtime, "max", max_length)

        # Narrow search area
        if runtime <= min_runtime:
            min_runtime = runtime
            min_length = length
        else:
            max_length = length

    if length <= max_length:
        if args.verbose:
            print("found for {}:".format(blocking_constant))
        print(length)
        sys.exit(0)
    else:
        if args.verbose:
            print("nothing found. exceeded search window and not change in performance found.")
        sys.exit(1)


def main():
    # Create and populate parser
    parser = create_parser()

    # Parse given arguments
    args = parser.parse_args()

    # BUSINESS LOGIC IS FOLLOWING
    run(parser, args)


if __name__ == '__main__':
    main()
