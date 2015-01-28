#!/usr/bin/env python

from __future__ import print_function

import argparse
import ast
import sys
import os.path
import subprocess
import re

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
    parser.add_argument('model', choices=['ECM', 'ECM-DATA', 'ECM-CPU'],
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
                            (3, 20*1024*1024, 'per socket', None)],
                           {'2': 'LOAD', '3': 'LOAD', '4': 'STORE'},
                           ['-xAVX'])
    
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
            
            if args.model in ['ECM', 'ECM-DATA']:
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
            if args.model in ['ECM', 'ECM-CPU']:
                # For the IACA/CPU analysis we need to compile and assemble
                asm_name = kernel.compile(compiler_args=machine.icc_flags)
                bin_name = kernel.assemble(asm_name, iaca_markers=True)
                
                iaca_output = subprocess.check_output(
                    ['iaca.sh', '-64', '-arch', machine.arch, bin_name])
                
                # Get total cycles per loop iteration
                match = re.search(
                    r'^Block Throughput: ([0-9\.]+) Cycles', iaca_output, re.MULTILINE)
                assert match, "Could not find Block Throughput in iaca output"
                block_throughput = match.groups()[0]
                
                # Find ports and cyles per port
                ports = filter(lambda l: l.startswith('|  Port  |'), iaca_output.split('\n'))
                cycles = filter(lambda l: l.startswith('| Cycles |'), iaca_output.split('\n'))
                assert ports and cycles, "Could not find ports/cylces lines in iaca output."
                ports = map(str.strip, ports[0].split('|'))[2:]
                cycles = map(str.strip, cycles[0].split('|'))[2:]
                port_cycles = []
                for i in range(len(ports)):
                    if '-' in ports[i] and ' ' in cycles[i]:
                        subports = map(str.strip, ports[i].split('-'))
                        subcycles = filter(bool, cycles[i].split(' '))
                        port_cycles.append((subports[0], subcycles[0]))
                        port_cycles.append((subports[0]+subports[1], subcycles[1]))
                    elif ports[i] and cycles[i]:
                        port_cycles.append((ports[i], cycles[i]))
                port_cycles = dict(port_cycles)
                
                print(machine.port_match)
                print('Ports and cycles:', port_cycles)
                print('Throughput:', block_throughput)

