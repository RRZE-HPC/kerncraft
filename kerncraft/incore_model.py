#!/usr/bin/env python3
"""Helper functions to instrument assembly code for and analyze with IACA."""
# Version check
import sys
import re
import subprocess
import os
from copy import copy
import argparse
from pprint import pformat, pprint
import textwrap
from collections import OrderedDict
import io

from distutils.spawn import find_executable
from osaca import osaca
from osaca.parser import get_parser
from osaca.semantics import MachineModel, ISASemantics
from osaca.semantics.marker_utils import find_basic_loop_bodies, get_marker

from kerncraft import iaca_get, __version__


def itemsEqual(lst):
   return lst[1:] == lst[:-1]


class IncoreModel:
    def __init__(self, isa='x86'):
        isa


class IACA(IncoreModel):
    pass


class OSACA(IncoreModel):
    pass


class LlvmMCA(IncoreModel):
    pass


class ISA:
    @staticmethod
    def get_isa(isa='x86'):
        if isa.lower() == 'x86':
            return x86
        elif isa.lower() == 'aarch64':
            return AArch64

    @staticmethod
    def compute_block_metric(block):
        """Compute sortable metric to rank blocks."""
        return NotImplementedError

    @classmethod
    def select_best_block(cls, blocks):
        """
        Return best block label selected based on simple heuristic.

        :param blocks: OrderedDict map of label to list of instructions
        """
        # TODO make this cleverer with more stats
        if not blocks:
            raise ValueError("No suitable blocks were found in assembly.")

        best_block_label = next(iter(blocks))
        best_metric = cls.compute_block_metric(blocks[best_block_label])

        for label, block in list(blocks.items())[1:]:
            metric = cls.compute_block_metric(block)
            if best_metric < metric:
                best_block_label = label
                best_metric = metric

        return best_block_label

    @staticmethod
    def get_pointer_increment(block):
        """Return pointer increment."""
        raise NotImplementedError


class x86(ISA):
    @staticmethod
    def compute_block_metric(block):
        """Return comparable metric on block information."""
        register_class_usage = {'zmm': [], 'ymm': [], 'xmm': []}
        packed_instruction_ctr, avx_instruction_ctr, instruction_ctr = 0, 0, 0
        # Analyze code to determine metric
        for line in block:
            # Skip non-instruction lines (e.g., comments)
            if line.instruction is None:
                continue
            # Count all instructions
            instruction_ctr += 1

            # Count registers used
            for prefix in register_class_usage:
                for op in line.operands:
                    if 'register' in op:
                        if op.register.name.startswith(prefix):
                            register_class_usage[prefix].append(op.register.name)

            # Identify and count packed and avx instructions
            if re.match(r"^[v]?(mul|add|sub|div|fmadd(132|213|231)?)[h]?p[ds]", line.instruction):
                if line.instruction.startswith('v'):
                    avx_instruction_ctr += 1
                packed_instruction_ctr += 1

        # Build metric
        return (instruction_ctr + packed_instruction_ctr + avx_instruction_ctr,
                len(set(register_class_usage['zmm'])),
                len(set(register_class_usage['ymm'])),
                len(set(register_class_usage['xmm'])))

    @staticmethod
    def get_pointer_increment(block):
        """Return pointer increment."""
        increments = {}
        mem_references = []
        stores_only = False
        for line in block:
            # Skip non-instruction lines (e.g., comments)
            if line.instruction is None:
                continue

            # Extract destination references
            dst_mem_references = [op.memory for op in line.semantic_operands.destination
                                  if 'memory' in op]
            if dst_mem_references:
                if not stores_only:
                    stores_only = True
                    mem_references = []
                mem_references += dst_mem_references

            # If no destination references were found sofar, include source references
            if not stores_only:
                mem_references += [op.memory for op in line.semantic_operands.source
                                  if 'memory' in op]

            if re.match(r'^inc[bwlq]?$', line.instruction):
                increments[line.operands[0].register.name] = 1
            elif re.match(r'^add[bwlq]?$', line.instruction) and 'immediate' in line.operands[0]:
                increments[line.operands[1].register.name] = int(line.operands[0].immediate.value)
            elif re.match(r'^dec[bwlq]?$', line.instruction):
                increments[[line.operands[0].register.name]] = -1
            elif re.match(r'^sub[bwlq]?$', line.instruction) and 'immediate' in line.operands[0]:
                increments[line.operands[1].register.name] = -int(line.operands[0].immediate.value)

        # deduce loop increment from memory index register
        pointer_increment = None  # default -> can not decide, let user choose
        possible_idx_regs = None
        if mem_references:
            # we found memory references to work with
            possible_idx_regs = list(set(increments.keys()).intersection(
                set([mref.base.name for mref in mem_references if mref.base is not None] +
                    [mref.index.name for mref in mem_references if mref.index is not None])))
            for mref in mem_references:
                for reg in list(possible_idx_regs):
                    # Only consider references with two registers, where one could be an
                    # index
                    if None not in [mref.base, mref.index]:
                        # One needs to mach, other registers will be excluded
                        if not ((mref.base is not None and reg == mref.base.name) or
                                (mref.index is not None and reg == mref.index.name)):
                            # reg can not be it
                            possible_idx_regs.remove(reg)

            idx_reg = None
            if len(possible_idx_regs) == 1:
                # good, exactly one register was found
                idx_reg = possible_idx_regs[0]
            elif possible_idx_regs and itemsEqual(
                    [increments[pidxreg] for pidxreg in possible_idx_regs]):
                # multiple were option found, but all have the same increment
                # use first match:
                idx_reg = possible_idx_regs[0]

            if idx_reg:
                mem_scales = [mref.scale for mref in mem_references
                              if (mref.index is not None and idx_reg == mref.index.name) or
                                 (mref.base is not None and idx_reg == mref.base.name)]

                if itemsEqual(mem_scales):
                    # good, all scales are equal
                    pointer_increment = mem_scales[0] * increments[idx_reg]

        return pointer_increment


class AArch64(ISA):
    @staticmethod
    def compute_block_metric(block):
        """Return comparable metric on block information."""
        instruction_ctr = 0
        # Analyze code to determine metric
        for line in block:
            # Skip non-instruction lines (e.g., comments)
            if line.instruction is None:
                continue
            # Count all instructions
            instruction_ctr += 1

        # Build metric
        return (instruction_ctr)

    @staticmethod
    def get_pointer_increment(block):
        """Return pointer increment."""
        return None


def userselect_increment(block):
    """Let user interactively select byte increment."""
    print("Selected block:")
    print('\n    ' + ('\n    '.join([b.line for b in block])))
    print()

    increment = None
    while increment is None:
        increment = input("Choose store pointer increment (number of bytes): ")
        try:
            increment = int(increment)
        except ValueError:
            increment = None
    return increment


def userselect_block(blocks, default=None, debug=False):
    """Let user interactively select block."""
    label_list = []
    print("Blocks found in assembly file:")
    for label, block in blocks.items():
        # Blocks first line is the label, the user will be able to spot it, so we don't need to
        # print it
        label_list.append(label)
        print('\n\t'.join([b['line'] for b in block]))

    # Show all possible block labels in the end
    print(
        '-----------------------------\n'
        + 'Possible blocks to be marked:'
    )
    for label in label_list:
        print(label)

    # Let user select block:
    block_label = None
    while block_label not in blocks:
        block_label = input("Choose block to be marked [" + str(default) + "]: ") or default

    return block_label


def parse_asm(code, isa):
    """Prase and process asm code."""
    asm_parser = get_parser(isa)
    asm_lines = asm_parser.parse_file(code)
    ISASemantics(isa).process(asm_lines)
    return asm_lines


def asm_instrumentation(input_file, output_file=None,
                        block_selection='auto',
                        pointer_increment='auto_with_manual_fallback',
                        debug=False,
                        isa='x86'):
    """
    Add markers to an assembly file.

    If instrumentation fails because loop increment could not determined automatically, a ValueError
    is raised.

    :param input_file: file-like object to read from
    :param output_file: file-like object to write to
    :param block_selection: label of the assembly block to instrument, or 'auto' for automatically
                            using block with the
                            most vector instructions, or 'manual' to read index to prompt user
    :param pointer_increment: number of bytes the pointer is incremented after the loop or
                              - 'auto': automatic detection, otherwise RuntimeError is raised
                              - 'auto_with_manual_fallback': like auto with fallback to manual input
                              - 'manual': prompt user
    :param debug: output additional internal analysis information. Only works with manual selection.
    :return: selected assembly block lines, pointer increment
    """
    asm_lines = parse_asm(input_file.read(), isa)

    # If input and output files are the same, overwrite with output
    if input_file is output_file:
        output_file.seek(0)
        output_file.truncate()

    if debug:
        block_selection = 'manual'

    loop_blocks = find_basic_loop_bodies(asm_lines)
    if block_selection == 'auto':
        block_label = ISA.get_isa(isa).select_best_block(loop_blocks)
    elif block_selection == 'manual':
        block_label = userselect_block(
            loop_blocks, default=ISA.get_isa(isa).select_best_block(loop_blocks), debug=debug)
    elif isinstance(block_selection, int):
        block_label = block_selection
    else:
        raise ValueError("block_selection has to be an integer, 'auto' or 'manual' ")
    block_lines = loop_blocks[block_label]

    block_start = asm_lines.index(block_lines[0])
    block_end = asm_lines.index(block_lines[-1]) + 1

    # Extract store pointer increment
    if not isinstance(pointer_increment, int):
        if pointer_increment == 'auto':
            pointer_increment = ISA.get_isa(isa).get_pointer_increment(block_lines)
            if pointer_increment is None:
                raise RuntimeError("pointer_increment could not be detected automatically. Use "
                                   "--pointer-increment to set manually to byte offset of store "
                                   "pointer address between consecutive assembly block iterations.")
        elif pointer_increment == 'auto_with_manual_fallback':
            pointer_increment = ISA.get_isa(isa).get_pointer_increment(block_lines)
            if pointer_increment is None:
                pointer_increment = userselect_increment(block_lines)
        elif pointer_increment == 'manual':
            pointer_increment = userselect_increment(block_lines)
        else:
            raise ValueError("pointer_increment has to be an integer, 'auto', 'manual' or  "
                             "'auto_with_manual_fallback' ")

    marker_start, marker_end = get_marker(
        isa, comment="pointer_increment={}".format(pointer_increment))

    marked_asm = asm_lines[:block_start] + marker_start + asm_lines[block_start:block_end] + \
                 marker_end + asm_lines[block_end:]

    if output_file is not None:
        output_file.writelines([l['line']+'\n' for l in marked_asm])

    return block_lines, pointer_increment


def osaca_analyse_instrumented_assembly(instrumented_assembly_file, micro_architecture):
    """
    Run OSACA analysis on an instrumented assembly.

    :param instrumented_assembly_file: path of assembly that was built with markers
    :param micro_architecture: micro architecture string as taken by OSACA.
                               one of: SNB, IVB, HSW, BDW, SKL
    :return: a dictionary with the following keys:
        - 'output': the output of the iaca executable
        - 'throughput': the block throughput in cycles for one possibly vectorized loop iteration
        - 'port cycles': dict, mapping port name to number of active cycles
        - 'uops': total number of Uops
    """
    result = {}
    isa = osaca.MachineModel.get_isa_for_arch(micro_architecture)
    parser = osaca.get_asm_parser(micro_architecture)
    with open(instrumented_assembly_file) as f:
        parsed_code = parser.parse_file(f.read())
    kernel = osaca.reduce_to_section(parsed_code, isa)
    osaca_machine_model = osaca.MachineModel(arch=micro_architecture)
    semantics = osaca.ArchSemantics(machine_model=osaca_machine_model)
    semantics.add_semantics(kernel)
    semantics.assign_optimal_throughput(kernel)

    kernel_graph = osaca.KernelDG(kernel, parser, osaca_machine_model)
    frontend = osaca.Frontend(instrumented_assembly_file, arch=micro_architecture)

    result['output'] = frontend.full_analysis(kernel, kernel_graph, verbose=True)
    throughput_values = semantics.get_throughput_sum(kernel)
    result['port cycles'] = OrderedDict(list(zip(osaca_machine_model['ports'], throughput_values)))
    result['throughput'] = max(semantics.get_throughput_sum(kernel))
    result['uops'] = None  # Not given by OSACA

    unmatched_ratio = osaca.get_unmatched_instruction_ratio(kernel)
    if unmatched_ratio > 0.1:
        print('WARNING: {:.0%} of the instruction could not be matched during incore analysis '
              'with OSACA. Fix this by extending OSACAs instruction form database with the '
              'required instructions.'.format(unmatched_ratio),
              file=sys.stderr)

    return result


def llvm_mca_analyse_instrumented_assembly(
        instrumented_assembly_file, micro_architecture, isa='x86'):
    """
    Run LLVM-MCA analysis on an instrumented assembly.

    :param instrumented_assembly_file: path of assembly that was built with markers
    :param micro_architecture: micro architecture string as taken by OSACA.
                               one of: SNB, IVB, HSW, BDW, SKL
    :return: a dictionary with the following keys:
        - 'output': the output of the iaca executable
        - 'throughput': the block throughput in cycles for one possibly vectorized loop iteration
        - 'port cycles': dict, mapping port name to number of active cycles
        - 'uops': total number of Uops
    """
    result = {}
    with open(instrumented_assembly_file) as f:
        parsed_code = parse_asm(f.read(), isa)
    kernel = osaca.reduce_to_section(parsed_code, isa)
    assembly_section = '\n'.join([l.line for l in kernel])

    output = subprocess.check_output(['llvm-mca']+micro_architecture.split(' '),
                                     input=assembly_section.encode('utf-8')).decode('utf-8')
    result['output'] = output

    # Extract port names
    port_names = OrderedDict()
    m = re.search(r'Resources:\n(?:[^\n]+\n)+', output)
    for m in re.finditer(r'(\[[0-9\.]+\])\s+-\s+([a-zA-Z0-9]+)', m.group()):
        port_names[m.group(1)] = m.group(2)

    # Extract cycles per port
    port_cycles = OrderedDict()
    m = re.search(r'Resource pressure per iteration:\n[^\n]+\n[^\n]+', output)
    port_cycle_lines = m.group().split('\n')[1:]
    for port, cycles in zip(port_cycle_lines[0].split(), port_cycle_lines[1].split()):
        if cycles == '-':
            cycles = 0.0
        port_cycles[port_names[port]] = float(cycles)

    result['port cycles'] = port_cycles
    result['throughput'] = max(port_cycles.values())

    # Extract uops
    uops = 0
    uops_raw = re.search(r'\n\[1\](\s+\[[0-9\.]+\]\s+)+Instructions:\n(:?\s*[0-9\.]+\s+[^\n]+\n)+',
                         output).group()
    for l in uops_raw.strip().split('\n')[2:]:
        uops += int(l.strip().split(' ')[0])

    result['uops'] = uops

    return result


def iaca_analyse_instrumented_binary(instrumented_binary_file, micro_architecture):
    """
    Run IACA analysis on an instrumented binary.

    :param instrumented_binary_file: path of binary that was built with IACA markers
    :param micro_architecture: micro architecture string as taken by IACA.
                               one of: NHM, WSM, SNB, IVB, HSW, BDW
    :return: a dictionary with the following keys:
        - 'output': the output of the iaca executable
        - 'throughput': the block throughput in cycles for one possibly vectorized loop iteration
        - 'port cycles': dict, mapping port name to number of active cycles
        - 'uops': total number of Uops
    """
    # Select IACA version and executable based on micro_architecture:
    arch_map = {
        # arch: (binary name, version string, required additional arguments)
        'NHM': ('iaca2.2', 'v2.2', ['-64']),
        'WSM': ('iaca2.2', 'v2.2', ['-64']),
        'SNB': ('iaca2.3', 'v2.3', ['-64']),
        'IVB': ('iaca2.3', 'v2.3', ['-64']),
        'HSW': ('iaca3.0', 'v3.0', []),
        'BDW': ('iaca3.0', 'v3.0', []),
        'SKL': ('iaca3.0', 'v3.0', []),
        'SKX': ('iaca3.0', 'v3.0', []),
    }

    if micro_architecture not in arch_map:
        raise ValueError('Invalid micro_architecture selected ({}), valid options are {}'.format(
            micro_architecture, ', '.join(arch_map.keys())))

    iaca_path = iaca_get.find_iaca()  # Throws exception if not found
    os.environ['PATH'] += ':' + iaca_path

    iaca_exec, iaca_version, base_args = arch_map[micro_architecture]
    if find_executable(iaca_exec) is None:
        raise RuntimeError("{0} executable was not found. Make sure that {0} is found in "
                           "{1}. Install using iaca_get.".format(iaca_exec, iaca_path))

    result = {}

    cmd = [iaca_exec] + base_args + ['-arch', micro_architecture, instrumented_binary_file]
    try:
        iaca_output = subprocess.check_output(cmd).decode('utf-8')
        result['output'] = iaca_output
    except OSError as e:
        raise RuntimeError("IACA execution failed:" + ' '.join(cmd) + '\n' + str(e))
    except subprocess.CalledProcessError as e:
        raise RuntimeError("IACA throughput analysis failed:" + str(e))

    # Get total cycles per loop iteration
    match = re.search(r'^Block Throughput: ([0-9.]+) Cycles', iaca_output, re.MULTILINE)
    assert match, "Could not find Block Throughput in IACA output."
    throughput = float(match.groups()[0])
    result['throughput'] = throughput

    # Find ports and cycles per port
    ports = [l for l in iaca_output.split('\n') if l.startswith('|  Port  |')]
    cycles = [l for l in iaca_output.split('\n') if l.startswith('| Cycles |')]
    assert ports and cycles, "Could not find ports/cycles lines in IACA output."
    ports = [p.strip() for p in ports[0].split('|')][2:]
    cycles = [c.strip() for c in cycles[0].split('|')][2:]
    port_cycles = []
    for i in range(len(ports)):
        if '-' in ports[i] and ' ' in cycles[i]:
            subports = [p.strip() for p in ports[i].split('-')]
            subcycles = [c for c in cycles[i].split(' ') if bool(c)]
            port_cycles.append((subports[0], float(subcycles[0])))
            port_cycles.append((subports[0] + subports[1], float(subcycles[1])))
        elif ports[i] and cycles[i]:
            port_cycles.append((ports[i], float(cycles[i])))
    result['port cycles'] = OrderedDict(port_cycles)

    match = re.search(r'^Total Num Of Uops: ([0-9]+)', iaca_output, re.MULTILINE)
    assert match, "Could not find Uops in IACA output."
    result['uops'] = float(match.groups()[0])
    return result


def main():
    """Execute command line interface."""
    parser = argparse.ArgumentParser(
        description='Find and analyze basic loop blocks and mark for IACA.',
        epilog='For help, examples, documentation and bug reports go to:\nhttps://github.com'
               '/RRZE-HPC/kerncraft\nLicense: AGPLv3')
    parser.add_argument('--version', action='version', version='%(prog)s {}'.format(__version__))
    parser.add_argument('source', type=argparse.FileType(), nargs='?', default=sys.stdin,
                        help='assembly file to analyze (default: stdin)')
    parser.add_argument('--outfile', '-o', type=argparse.FileType('w'), nargs='?',
                        default=sys.stdout, help='output file location (default: stdout)')
    parser.add_argument('--debug', action='store_true',
                        help='Output internal analysis information for debugging.')
    parser.add_argument('--isa', default='x86', choices=['x86', 'aarch64'])
    args = parser.parse_args()

    # pointer_increment is given, since it makes no difference on the command lien and requires
    # less user input
    asm_instrumentation(input_file=args.source, output_file=args.outfile,
                        block_selection='manual', pointer_increment='auto_with_manual_fallback',
                        debug=args.debug, isa=args.isa)


if __name__ == '__main__':
    main()
