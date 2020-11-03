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
from collections import OrderedDict, defaultdict
import io
from hashlib import md5
from os.path import expanduser
from itertools import chain

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
            if re.match(r"^[v]?(movu|mul|add|sub|div|fmadd(132|213|231)?)[h]?p[ds]",
                    line.instruction):
                if line.instruction.startswith('v'):
                    avx_instruction_ctr += 1
                packed_instruction_ctr += 1

        # Build metric
        return (packed_instruction_ctr, avx_instruction_ctr, instruction_ctr,
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

            # Extract destination references, ignoring var(%rip)
            dst_mem_references = [op.memory for op in line.semantic_operands.destination
                                  if 'memory' in op and
                                  (op.memory.base is None or op.memory.base.name != 'rip')]
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
            elif re.match(r'^add[bwlq]?$', line.instruction) and 'immediate' in line.operands[0] \
                    and 'register' in line.operands[1]:
                increments[line.operands[1].register.name] = int(line.operands[0].immediate.value)
            elif re.match(r'^dec[bwlq]?$', line.instruction):
                increments[line.operands[0].register.name] = -1
            elif re.match(r'^sub[bwlq]?$', line.instruction) and 'immediate' in line.operands[0] \
                    and 'register' in line.operands[1]:
                increments[line.operands[1].register.name] = -int(line.operands[0].immediate.value)
            elif re.match(r'^lea[bwlq]?$', line.instruction):
                # `lea 1(%r11), %r11` is the same as `add $1, %r11`
                if line.operands[0].memory.base.name  == line.operands[1].register.name and \
                        line.operands[0].memory.index is None:
                    increments[line.operands[1].register.name] = int(
                        line.operands[0].memory.offset.value)


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

        if pointer_increment is None:
            pointer_increment = find_increment_in_cache(block)
        return pointer_increment


class AArch64(ISA):
    @staticmethod
    def compute_block_metric(block):
        """Return comparable metric on block information."""
        arithmetic_ctr = 0
        vector_ctr = 0
        instruction_ctr = 0
        # Analyze code to determine metric
        for line in block:
            # Skip non-instruction lines (e.g., comments)
            if line.instruction is None:
                continue
            # Counting basic arithmetic insstructions
            if line.instruction in ['add', 'sub', 'mul', 'fmul', 'fdiv', 'fadd', 'fsub']:
                arithmetic_ctr += 1
            # Counting use of vector registers
            for op in line.operands:
                if 'register' in op and op.register.prefix in 'zv':
                    vector_ctr += 1
            # Count all instructions
            instruction_ctr += 1

        # Build metric
        return (vector_ctr, arithmetic_ctr, instruction_ctr)
    
    @staticmethod
    def normalize_to_register_str(register):
        if register is None:
            return None
        prefix = register.prefix
        if prefix in 'wx':
            prefix = 'x'
        return prefix + register.name

    @staticmethod
    def get_pointer_increment(block):
        """Return pointer increment."""
        increments = defaultdict(int)
        mem_references = []
        stores_only = False

        # build dict of modified registers in block with count of number of modifications
        modified_registers = defaultdict(int)
        for dests in [l.semantic_operands.destination for l in block if 'semantic_operands' in l]:
            for d in dests:
                if 'register' in d:
                    modified_registers[AArch64.normalize_to_register_str(d.register)] += 1
        for l in block:
            for d in l.operands:
                if 'memory' in d:
                    if 'post_indexed' in d.memory or 'pre_indexed' in d.memory:
                        modified_registers[AArch64.normalize_to_register_str(d.memory.base)] += 1
                        inc = 1
                        if 'post_indexed' in d.memory and 'value' in d.memory.post_indexed:
                            inc = int(d.memory.post_indexed.value)
                        if 'pre_indexed' in d.memory:
                            inc = int(d.memory.offset.value)
                        increments[AArch64.normalize_to_register_str(d.memory.base)] = inc
        
        for line in block:
            # Skip non-instruction lines (such as comments and labels)
            if line.instruction is None:
                continue

            # Extract and filter destination references (stores)
            dst_mem_references = []
            for dst in [op.memory for op in chain(line.semantic_operands.destination,
                                                  line.semantic_operands.src_dst)
                        if 'memory' in op]:
                # base or index must be a modified (i.e., changing) register
                if AArch64.normalize_to_register_str(dst.base) not in modified_registers and \
                    AArch64.normalize_to_register_str(dst.index) not in modified_registers:
                    continue

                # offset operands with identifiers (e.g. `:lo12:gosa`) are ignored
                if dst.offset is not None and 'identifier' in dst.offset:
                    continue

                dst_mem_references.append(dst)
            # consider only stores from here on
            if dst_mem_references and not stores_only:
                stores_only = True
                mem_references = []
            mem_references += dst_mem_references

            # If no destination references were found sofar, include source references (loads)
            if not stores_only:
                mem_references += [op.memory for op in chain(line.semantic_operands.source,
                                                             line.semantic_operands.src_dst)
                                   if 'memory' in op]

            # ADD dest_reg, src_reg, immd
            if re.match(r'^add[s]?$', line.instruction) and \
                    line.operands[0] == line.operands[1] and \
                    'immediate' in line.operands[2]:
                reg_name = AArch64.normalize_to_register_str(line.operands[0].register)
                inc = int(line.operands[2].immediate.value)
                increments[reg_name] = inc
            # SUB dest_reg, src_reg, immd
            elif re.match(r'^sub[s]?$', line.instruction) and \
                    line.operands[0] == line.operands[1] and \
                    'immediate' in line.operands[2]:
                reg_name = AArch64.normalize_to_register_str(line.operands[0].register)
                inc = -int(line.operands[2].immediate.value)
                if reg_name in increments and increments[reg_name] == inc:
                    increments[reg_name] = inc

        # Remove any increments that are modiefed more than once
        increments = {reg_name: inc for reg_name, inc in increments.items()
                      if modified_registers[reg_name] == 1}

        # Second pass to find lsl instructions on increments
        for line in block:
            if line.instruction is None:
                continue
            # LSL dest_reg, src_reg, immd
            if re.match(r'^lsl$', line.instruction) and 'immediate' in line.operands[2] and \
                    AArch64.normalize_to_register_str(line.operands[1].register) in increments:
                increments[AArch64.normalize_to_register_str(line.operands[0].register)] = \
                    increments[AArch64.normalize_to_register_str(line.operands[1].register)] * \
                    2**int(line.operands[2].immediate.value)

        new_increments = []
        # Third pass to find registers based on constant +- increment
        for line in block:
            if line.instruction is None:
                continue
            # ADD|SUB dest_reg, const_reg, increment_reg (source registers may be switched)
            m = re.match(r'^(add|sub)[s]?$', line.instruction)
            if m:
                if m.group(1) == 'add':
                    factor = 1
                else:
                    factor = -1
                if 'register' not in line.operands[1] or 'register' not in line.operands[2]:
                    continue
                for i,j in [(1,2), (2,1)]:
                    reg_i_name = AArch64.normalize_to_register_str(line.operands[i].register)
                    reg_j_name = AArch64.normalize_to_register_str(line.operands[j].register)
                    if reg_i_name in increments and reg_j_name not in modified_registers:
                        reg_dest_name = AArch64.normalize_to_register_str(line.operands[0].register)
                        inc = factor * increments[reg_i_name]
                        if reg_dest_name in increments and increments[reg_dest_name] == inc:
                            modified_registers[reg_dest_name] -= 1
                        increments[reg_dest_name] = inc
                        new_increments.append(reg_dest_name)

        # Remove any increments that are modified more often than updates have been detected
        increments = {reg_name: inc for reg_name, inc in increments.items()
                      if modified_registers[reg_name] == 1}

        # Last pass to find lsl instructions on increments
        for line in block:
            if line.instruction is None:
                continue
            # LSL dest_reg, src_reg, immd
            if re.match(r'^lsl$', line.instruction) and 'immediate' in line.operands[2] and \
                    'register' in line.operands[1]:
                src_reg_name = AArch64.normalize_to_register_str(line.operands[1].register)
                if src_reg_name in new_increments and src_reg_name in increments:
                    increments[AArch64.normalize_to_register_str(line.operands[0].register)] = \
                        increments[src_reg_name] * \
                        2**int(line.operands[2].immediate.value)

        # deduce loop increment from memory index register
        address_registers = []
        scales = defaultdict(lambda: 1)
        for mref in mem_references:
            # Assume base to be scaled
            base_reg = AArch64.normalize_to_register_str(mref.base)
            if mref.index is not None:
                # If index register is used, check which is incremented
                index_reg = AArch64.normalize_to_register_str(mref.index)
                if index_reg in increments:
                    reg = index_reg
                    # If index is used, a scale other than 1 needs to be considered
                    if 'shift' in mref.index:
                        scales[reg] = 2**int(mref.index.shift.value)
                else:
                    reg = base_reg
            else:
                reg = base_reg
            # ignore all unmodiefed registers
            if reg in modified_registers:
                address_registers.append(reg)
            increment = None

        pointer_increment = None  # default -> can not decide, let user choose
        if address_registers and all([reg in increments for reg in address_registers]):
            if itemsEqual([increments[reg] for reg in address_registers]):
                # good, all relevant increments are equal
                pointer_increment = increments[address_registers[0]] * scales[address_registers[0]]

        # Check cache as last resort
        if pointer_increment is None:
            pointer_increment = find_increment_in_cache(block)

        return pointer_increment


def userselect_increment(block, default=None):
    """Let user interactively select byte increment."""
    print("Selected block:")
    print('\n    ' + ('\n    '.join([b.line for b in block])))
    print('hash: ', hashblock(block))
    print()

    increment = None
    while increment is None:
        prompt = "Choose store pointer increment (number of bytes)"
        if default:
            prompt += '[{}]'.format(default)
        prompt += ': '
        increment = input(prompt)
        try:
            increment = int(increment)
        except ValueError:
            increment = default
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


def hashblock(block):
    """Hashes block down to relevant info"""
    # TODO normalize register names
    # TODO normalize instruction order
    # Remove target label and jump
    h = md5('\n'.join([b['line'] for b in block]).encode())
    return h.hexdigest()


def find_increment_in_cache(block, cache_file='~/.kerncraft/increment_cache'):
    search_hash = hashblock(block)
    cache_file = expanduser(cache_file)
    cache = ''
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            cache = f.readlines()
    for c in cache:
        c_split = c.split()
        if len(c_split) != 2:
            continue
        hashstr, increment = c_split
        try:
            increment = int(increment)
        except:
            increment = None
        if hashstr == search_hash:
            return increment
    return None


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
                              - 'auto_with_cache_fallback': like auto with fallback to cache file
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
    block_hashstr = hashblock(block_lines)

    block_start = asm_lines.index(block_lines[0])
    block_end = asm_lines.index(block_lines[-1]) + 1

    # Extract store pointer increment
    if not isinstance(pointer_increment, int):
        if pointer_increment == 'auto':
            pointer_increment = ISA.get_isa(isa).get_pointer_increment(block_lines)
            if pointer_increment is None:
                if output_file is not None:
                    os.unlink(output_file.name)
                raise RuntimeError("pointer_increment could not be detected automatically. Use "
                                   "--pointer-increment to set manually to byte offset of store "
                                   "pointer address between consecutive assembly block iterations. "
                                   "Alternativley add the following line to ~/.kerncraft/"
                                   "increment_cache: {} <pointer_increment>".format(block_hashstr))
        elif pointer_increment == 'auto_with_manual_fallback':
            pointer_increment = ISA.get_isa(isa).get_pointer_increment(block_lines)
            if pointer_increment is None:
                pointer_increment = userselect_increment(block_lines)
        elif pointer_increment == 'manual':
            pointer_increment = ISA.get_isa(isa).get_pointer_increment(block_lines)
            pointer_increment = userselect_increment(block_lines, default=pointer_increment)
        else:
            raise ValueError("pointer_increment has to be an integer, 'auto', 'manual' or  "
                             "'auto_with_manual_fallback' ")

    marker_start, marker_end = get_marker(
        isa, comment="pointer_increment={} {}".format(pointer_increment, block_hashstr))

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

    # Throughput Analysis
    throughput_values = semantics.get_throughput_sum(kernel)
    # LCD Latency Analysis
    lcd_dict = kernel_graph.get_loopcarried_dependencies()
    max_lcd = 0
    for dep in lcd_dict:
        max_lcd = max(
            max_lcd,
            sum([instr_form['latency_lcd'] for instr_form in lcd_dict[dep]['dependencies']]))

    result['output'] = frontend.full_analysis(kernel, kernel_graph, verbose=True)
    result['port cycles'] = OrderedDict(list(zip(osaca_machine_model['ports'], throughput_values)))
    result['throughput'] = max(throughput_values + [max_lcd])
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

    output = subprocess.check_output(
        ['llvm-mca'] + micro_architecture.split(' ') + 
        ['--timeline', '--timeline-max-cycles=1000', '--timeline-max-iterations=4'],
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
        if port_names[port] in port_cycles:
            # Some architecures have multiple "ports" per resource in LLVM-MCA
            # e.g., Sandybridge as a Port23 resource which is found at [6.0] and [6.1]
            # we will consider the maximum of both
            port_cycles[port_names[port]] = max(float(cycles), port_cycles[port_names[port]])
        else:
            port_cycles[port_names[port]] = float(cycles)
    result['port cycles'] = port_cycles
    
    # Extract throughput including loop-carried-dependecy latency
    timeline_lines = [l for l in output.split('\n') if re.match(r'\[[0-9]+,[0-9]+\]', l)]
    lcd = 0
    for l in timeline_lines:
        if l.startswith('[0,'):
            last_instr_index = re.match(r'\[0,([0-9]+)\]', l).group(1)
            lcd_start = l.index('R')
        elif l.startswith('[1,'+last_instr_index+']'):
            lcd = l.index('R') - lcd_start
            break
    result['throughput'] = max(max(port_cycles.values()), lcd)

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
