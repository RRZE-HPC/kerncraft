#!/usr/bin/env python
"""Helper functions to instumentalize assembly code for and analyze with IACA."""
from __future__ import print_function
from __future__ import absolute_import

# Version check
import sys
if sys.version_info[0] == 2 and sys.version_info < (2, 7) or \
        sys.version_info[0] == 3 and sys.version_info < (3, 4):
    print("Must use python 2.7 or 3.4 and greater.", file=sys.stderr)
    sys.exit(1)

import re
import subprocess
import os
from copy import copy

from distutils.spawn import find_executable
from six.moves import input

from kerncraft import iaca_get


# Within loop
START_MARKER = ['        movl      $111, %ebx # INSERTED BY KERNCRAFT IACA MARKER UTILITY\n'
                '        .byte     100        # INSERTED BY KERNCRAFT IACA MARKER UTILITY\n'
                '        .byte     103        # INSERTED BY KERNCRAFT IACA MARKER UTILITY\n'
                '        .byte     144        # INSERTED BY KERNCRAFT IACA MARKER UTILITY\n']
# After loop
END_MARKER = ['        movl      $222, %ebx # INSERTED BY KERNCRAFT IACA MARKER UTILITY\n'
              '        .byte     100        # INSERTED BY KERNCRAFT IACA MARKER UTILITY\n'
              '        .byte     103        # INSERTED BY KERNCRAFT IACA MARKER UTILITY\n'
              '        .byte     144        # INSERTED BY KERNCRAFT IACA MARKER UTILITY\n']


def strip_and_uncomment(asm_lines):
    """Strip whitespaces and comments from asm lines."""
    asm_stripped = []
    for line in asm_lines:
        # Strip comments and whitespaces
        asm_stripped.append(line.split('#')[0].strip())
    return asm_stripped


def strip_unreferenced_labels(asm_lines):
    """Strip all labels, which are never referenced."""
    asm_code = '\n'.join(asm_lines)  # Needed for search of references
    asm_stripped = []
    for line in asm_lines:
        if re.match(r'^\S+:', line):
            # Found label
            label = line[0:line.find(':')]
            # Search for references to current label
            if not re.search('\s'+re.escape(label)+'[\s,]?.*$', asm_code, re.MULTILINE):
                # Skip labels without seen reference
                line = ''
        asm_stripped.append(line)
    return asm_stripped


def find_asm_blocks(asm_lines):
    """Find blocks probably corresponding to loops in assembly."""
    blocks = []

    last_label_line = -1
    last_label = None
    packed_ctr = 0
    avx_ctr = 0
    xmm_references = []
    ymm_references = []
    gp_references = []
    mem_references = []
    increments = {}
    for i, line in enumerate(asm_lines):
        # Register access counts
        ymm_references += re.findall('%ymm[0-9]+', line)
        xmm_references += re.findall('%xmm[0-9]+', line)
        gp_references += re.findall('%r[a-z0-9]+', line)

        if re.search(r'\d*\(%\w+,%\w+(,\d)?\)', line):
            m = re.search(r'(?P<off>\d*)\(%(?P<basep>\w+),%(?P<idx>\w+)(?:,(?P<scale>\d))?\)',
                          line)
            mem_references.append((
                int(m.group('off')) if m.group('off') else 0,
                m.group('basep'),
                m.group('idx'),
                int(m.group('scale')) if m.group('scale') else 1))

        if re.match(r"^[v]?(mul|add|sub|div)[h]?p[ds]", line):
            if line.startswith('v'):
                avx_ctr += 1
            packed_ctr += 1
        elif re.match(r'^\S+:', line):
            last_label = line[0:line.find(':')]
            last_label_line = i

            # Reset counters
            packed_ctr = 0
            avx_ctr = 0
            xmm_references = []
            ymm_references = []
            gp_references = []
            mem_references = []
            increments = {}
        elif re.match(r'^inc[bwlq]?\s+%\[a-z0-9]+', line):
            reg_start = line.find('%')+1
            increments[line[reg_start:]] = 1
        elif re.match(r'^add[bwlq]?\s+\$[0-9]+,\s*%[a-z0-9]+', line):
            const_start = line.find('$')+1
            const_end = line[const_start+1:].find(',')+const_start+1
            reg_start = line.find('%')+1
            increments[line[reg_start:]] = int(line[const_start:const_end])
        elif re.match(r'^dec[bwlq]?', line):
            reg_start = line.find('%')+1
            increments[line[reg_start:]] = -1
        elif re.match(r'^sub[bwlq]?\s+\$[0-9]+,', line):
            const_start = line.find('$')+1
            const_end = line[const_start+1:].find(',')+const_start+1
            reg_start = line.find('%')+1
            increments[line[reg_start:]] = -int(line[const_start:const_end])
        elif last_label and re.match(r'^j[a-z]+\s+'+re.escape(last_label)+r'\s*', line):
            # End of block
            # deduce loop increment from memory index register
            pointer_increment = None  # default -> can not decide, let user choose
            possible_idx_regs = None
            if mem_references:
                # we found memory references to work with
                possible_idx_regs = list(increments.keys())
                for mref in mem_references:
                    for reg in possible_idx_regs:
                        if not (reg == mref[1] or reg == mref[2]):
                            # reg can not be it
                            possible_idx_regs.remove(reg)
                            break

                if len(possible_idx_regs) == 1:
                    # good, exactly one register was found
                    idx_reg = possible_idx_regs[0]

                    mem_scales = [mref[3] for mref in mem_references
                                  if idx_reg == mref[2] or idx_reg == mref[1]]

                    if mem_scales[1:] == mem_scales[:-1]:
                        # good, all scales are equal
                        try:
                            pointer_increment = mem_scales[0]*increments[idx_reg]
                        except:
                            print("label", last_label)
                            print("lines", repr(asm_lines[last_label_line:i+1]))
                            print("increments", increments)
                            print("mem_references", mem_references)
                            print("idx_reg", idx_reg)
                            print("mem_scales", mem_scales)
                            raise

            blocks.append({'first_line': last_label_line,
                           'last_line': i,
                           'ops': i-last_label_line,
                           'label': last_label,
                           'packed_instr': packed_ctr,
                           'avx_instr': avx_ctr,
                           'XMM': (len(xmm_references), len(set(xmm_references))),
                           'YMM': (len(ymm_references), len(set(ymm_references))),
                           'GP': (len(gp_references), len(set(gp_references))),
                           'regs': (len(xmm_references) + len(ymm_references) + len(gp_references),
                                    len(set(xmm_references)) + len(set(ymm_references)) +
                                    len(set(gp_references))),
                           'pointer_increment': pointer_increment,
                           'lines': asm_lines[last_label_line:i+1],
                           'possible_idx_regs': possible_idx_regs,
                           'mem_references': mem_references,
                           'increments': increments,})

            # Reset counters
            packed_ctr = 0
            avx_ctr = 0
            xmm_references = []
            ymm_references = []
            gp_references = []
            mem_references = []
            increments = {}
    return list(enumerate(blocks))


def select_best_block(blocks):
    """Return best block selected based on simple heuristic."""
    # TODO make this cleverer with more stats
    best_block = max(blocks, key=lambda b: b[1]['packed_instr'])
    if best_block[1]['packed_instr'] == 0:
        best_block = max(blocks, key=lambda b: b[1]['ops']+b[1]['packed_instr']+b[1]['avx_instr'])
    return best_block[0]


def userselect_increment(block):
    """Let user interactively select byte increment."""
    print("Selected block:")
    print('\n    '+('    '.join(block['lines'])))
    print()

    increment = None
    while increment is None:
        increment = input("Choose store pointer increment (number of bytes): ")
        try:
            increment = int(increment)
        except ValueError:
            increment = None

    block['pointer_increment'] = increment
    return increment


def userselect_block(blocks, default=None):
    """Let user interactively select block."""
    print("Blocks found in assembly file:")
    print("   block   | OPs | pck. | AVX || Registers |    YMM   |    XMM   |    GP   ||ptr.inc|\n"
          "-----------+-----+------+-----++-----------+----------+----------+---------++-------|")
    for idx, b in blocks:
        print('{:>2} {b[label]:>7} | {b[ops]:>3} | {b[packed_instr]:>4} | {b[avx_instr]:>3} |'
              '| {b[regs][0]:>3} ({b[regs][1]:>3}) | {b[YMM][0]:>3} ({b[YMM][1]:>2}) | '
              '{b[XMM][0]:>3} ({b[XMM][1]:>2}) | {b[GP][0]:>2} ({b[GP][1]:>2}) || '
              '{b[pointer_increment]!s:>5} |'.format(idx, b=b))

    # Let user select block:
    block_idx = -1
    while not (0 <= block_idx < len(blocks)):
        block_idx = input("Choose block to be marked ["+str(default)+"]: ") or default
        try:
            block_idx = int(block_idx)
        except ValueError:
            block_idx = -1
    # block = blocks[block_idx][1]

    return block_idx


def insert_markers(asm_lines, start_line, end_line):
    """Insert IACA marker into list of ASM instructions at given indices."""
    asm_lines = asm_lines[:start_line] + START_MARKER + \
        asm_lines[start_line:end_line+1] + END_MARKER + \
        asm_lines[end_line+1:]
    return asm_lines


def iaca_instrumentation(input_file, output_file=None,
                         block_selection='auto',
                         pointer_increment='auto_with_manual_fallback'):
    """
    Add IACA markers to an assembly file.

    If instrumentation fails because loop increment could not determined automatically, a ValueError
    is raised.

    :param input_file: path to assembly file used as input
    :param output_file: output path, if None the input is overwritten
    :param block_selection: index of the assembly block to instrument, or 'auto' for automatically
                            using block with the
                            most vector instructions, or 'manual' to read index to prompt user
    :param pointer_increment: number of bytes the pointer is incremented after the loop or
                              - 'auto': automatic detection, otherwise RuntimeError is raised
                              - 'auto_with_manual_fallback': like auto with fallback to manual input
                              - 'manual': prompt user
    :return: the instrumented assembly block
    """
    if output_file is None:
        output_file = input_file

    with open(input_file, 'r') as f:
        assembly_orig = f.readlines()

    assembly = strip_and_uncomment(copy(assembly_orig))
    assembly = strip_unreferenced_labels(assembly)
    blocks = find_asm_blocks(assembly)
    if block_selection == 'auto':
        block_idx = select_best_block(blocks)
    elif block_selection == 'manual':
        block_idx = userselect_block(blocks, default=select_best_block(blocks))
    elif isinstance(block_selection, int):
        block_idx = block_selection
    else:
        raise ValueError("block_selection has to be an integer, 'auto' or 'manual' ")

    block = blocks[block_idx][1]

    if pointer_increment == 'auto':
        if block['pointer_increment'] is None:
            raise RuntimeError("pointer_increment could not be detected automatically")
    elif pointer_increment == 'auto_with_manual_fallback':
        if block['pointer_increment'] is None:
            block['pointer_increment'] = userselect_increment(block)
    elif pointer_increment == 'manual':
        block['pointer_increment'] = userselect_increment(block)
    elif isinstance(pointer_increment, int):
        block['pointer_increment'] = pointer_increment
    else:
        raise ValueError("pointer_increment has to be an integer, 'auto', 'manual' or  "
                         "'auto_with_manual_fallback' ")

    instrumentedAsm = insert_markers(assembly_orig, block['first_line'], block['last_line'])
    with open(output_file, 'w') as in_file:
        in_file.writelines(instrumentedAsm)

    return block


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
        raise RuntimeError("{} executable was not found. Make sure that {} is found in "
                               "{}. Install using iaca_get.".format(iaca_exec, iaca_path))

    result = {}

    try:
        cmd = [iaca_exec] + base_args + ['-arch', micro_architecture, instrumented_binary_file]
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
    result['port cycles'] = dict(port_cycles)

    match = re.search(r'^Total Num Of Uops: ([0-9]+)', iaca_output, re.MULTILINE)
    assert match, "Could not find Uops in IACA output."
    result['uops'] = float(match.groups()[0])
    return result


def main():
    """Execute command line interface."""
    if len(sys.argv) != 2:
        print("Usage:", sys.argv[0], "filename.s")
        sys.exit(1)

    iaca_instrumentation(input_file=sys.argv[1], output_file=sys.argv[1],
                         block_selection='manual', pointer_increment=1)

    print("Markers inserted.")


if __name__ == '__main__':
    main()
