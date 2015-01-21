#!/usr/bin/env python

from __future__ import print_function

import argparse
import ast
import sys
import os.path

from ecm import ECM
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cpu', default='Xeon E5-2680',
                        help='CPU model to be analized for (default "Xeon E5-2680")')
    parser.add_argument('model', choices=['ECM'],
                        help='Performance model to apply')
    parser.add_argument('code_file', type=argparse.FileType(), nargs='+',
                        help='File with loop kernel C code')
    parser.add_argument('-D', '--define', nargs=2, metavar=('KEY', 'VALUE'), default=[],
                        action=AppendStringInteger,
                        help='Define constant to be used in C code. Values must be integers. ' + \
                             'Overwrites constants from testcase file.')
    parser.add_argument('--testcases', '-t', action='store_true',
                        help='Use testcases file.')
    parser.add_argument('--testcase-index', '-i', metavar='INDEX', type=int, default=0,
                        help='Index of testcase in testcase file. If not given, all cases are ' + \
                             'executed.')
    
    # BUSINESS LOGIC IS FOLLOWING
    args = parser.parse_args()
    
    # machine information
    # Read machine description
    machine = {
        'name': 'Intel Xeon 2660v2',
        'clock': '2.2 GHz',
        'IACA architecture': 'IVB',
        'caheline': '64 B',
        'memory bandwidth': '60 GB/s',
        'cache stack': 
            [{'level': 1, 'size': '32 KB', 'type': 'per core', 'bw': '2 cy/CL'},
             {'level': 2, 'size': '256 KB', 'type': 'per core', 'bw': '2 cy/CL'},
             {'level': 3, 'size': '25 MB', 'type': 'per socket'}]
    }
    # TODO support format as seen above
    # TODO missing in description bw_type, size_type, read and write bw between levels
    #      and cache size sharing and cache bw sharing
    #machine = MachineModel('Intel Xeon 2660v2', 'IVB', 2.2e9, 10, 64, 60e9, 
    #                       [(1, 32*1024, 'per core', 2),
    #                        (2, 256*1024, 'per core', 2),
    #                        (3, 25*1024*1024, 'per socket', None)])
    # SNB machine as described in ipdps15-ECM.pdf
    machine = MachineModel('Intel Xeon E5-2680', 'SNB', 2.7e9, 8, 64, 40e9,
                           [(1, 32*1024, 'per core', 2),
                            (2, 256*1024, 'per core', 2),
                            (3, 20*1024*1024, 'per socket', None)])
    
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
        
            kernel = Kernel(code)
            
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
            
            if args.model == 'ECM':
                # Analyze access patterns (in regard to cache sizes with layer conditions)
                ecm = ECM(kernel, None, machine)
                results = ecm.calculate_cache_access()  # <-- this is my thesis
                if 'results-to-compare' in testcase:
                    for key, value in results.items():
                        if key in testcase['results-to-compare']:
                            correct_value = testcase['results-to-compare'][key]
                            diff = abs(value - correct_value)
                            if diff > correct_value*0.1:
                                print("Test values did not match: {} ".format(key) +
                                    "should have been {}, but was {}.".format(correct_value, value))
                                sys.exit(1)
                            elif diff:
                                print("Small difference from theoretical value: {} ".format(key) +
                                    "should have been {}, but was {}.".format(correct_value, value))