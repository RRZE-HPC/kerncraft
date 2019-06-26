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

from distutils.spawn import find_executable
from osaca.osaca import OSACA, extract_marked_section

from kerncraft import iaca_get
from . import __version__

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
    asm_stripped = []
    for line in asm_lines:
        if re.match(r'^\S+:', line):
            # Found label
            label = line[0:line.find(':')]
            # Search for references to current label
            if not any([re.match(r'^[^#]*\s' + re.escape(label) + '[\s,]?.*$', l)
                        for l in asm_lines]):
                # Skip labels without seen reference
                line = ''
        asm_stripped.append(line)
    return asm_stripped


def itemsEqual(lst):
   return lst[1:] == lst[:-1]


def find_asm_blocks(asm_lines):
    """Find blocks probably corresponding to loops in assembly."""
    blocks = []

    last_labels = OrderedDict()
    packed_ctr = 0
    avx_ctr = 0
    xmm_references = []
    ymm_references = []
    zmm_references = []
    gp_references = []
    mem_references = []
    increments = {}
    for i, line in enumerate(asm_lines):
        # Register access counts
        zmm_references += re.findall('%zmm[0-9]+', line)
        ymm_references += re.findall('%ymm[0-9]+', line)
        xmm_references += re.findall('%xmm[0-9]+', line)
        gp_references += re.findall('%r[a-z0-9]+', line)
        if re.search(r'\d*\(%\w+(,%\w+)?(,\d)?\)', line):
            m = re.search(r'(?P<off>[-]?\d*)\(%(?P<basep>\w+)(,%(?P<idx>\w+))?(?:,(?P<scale>\d))?\)'
                          r'(?P<eol>$)?',
                          line)
            mem_references.append((
                int(m.group('off')) if m.group('off') else 0,
                m.group('basep'),
                m.group('idx'),
                int(m.group('scale')) if m.group('scale') else 1,
                'load' if m.group('eol') is None else 'store'))

        if re.match(r"^[v]?(mul|add|sub|div|fmadd(132|213|231)?)[h]?p[ds]", line):
            if line.startswith('v'):
                avx_ctr += 1
            packed_ctr += 1
        elif re.match(r'^\S+:', line):
            # last_labels[label_name] = line_number
            last_labels[line[0:line.find(':')]] =i

            # Reset counters
            packed_ctr = 0
            avx_ctr = 0
            xmm_references = []
            ymm_references = []
            zmm_references = []
            gp_references = []
            mem_references = []
            increments = {}
        elif re.match(r'^inc[bwlq]?\s+%[a-z0-9]+', line):
            reg_start = line.find('%') + 1
            increments[line[reg_start:]] = 1
        elif re.match(r'^add[bwlq]?\s+\$[0-9]+,\s*%[a-z0-9]+', line):
            const_start = line.find('$') + 1
            const_end = line[const_start + 1:].find(',') + const_start + 1
            reg_start = line.find('%') + 1
            increments[line[reg_start:]] = int(line[const_start:const_end])
        elif re.match(r'^dec[bwlq]?', line):
            reg_start = line.find('%') + 1
            increments[line[reg_start:]] = -1
        elif re.match(r'^sub[bwlq]?\s+\$[0-9]+,', line):
            const_start = line.find('$') + 1
            const_end = line[const_start + 1:].find(',') + const_start + 1
            reg_start = line.find('%') + 1
            increments[line[reg_start:]] = -int(line[const_start:const_end])
        elif last_labels and re.match(r'^j[a-z]+\s+\S+\s*', line):
            # End of block(s) due to jump

            # Check if jump target matches any previously recoded label
            last_label = None
            last_label_line = -1
            for label_name, label_line in last_labels.items():
                if re.match(r'^j[a-z]+\s+' + re.escape(label_name) + r'\s*', line):
                    # matched
                    last_label = label_name
                    last_label_line = label_line

            labels = list(last_labels.keys())

            if last_label:
                # deduce loop increment from memory index register
                pointer_increment = None  # default -> can not decide, let user choose
                possible_idx_regs = None
                if mem_references:
                    # we found memory references to work with

                    # If store accesses exist, consider only those
                    store_references = [mref for mref in mem_references
                                        if mref[4] == 'store']
                    refs = store_references or mem_references

                    possible_idx_regs = list(set(increments.keys()).intersection(
                        set([r[1] for r in refs if r[1] is not None] +
                            [r[2] for r in refs if r[2] is not None])))
                    for mref in refs:
                        for reg in list(possible_idx_regs):
                            # Only consider references with two registers, where one could be an
                            # index
                            if None not in mref[1:3]:
                                # One needs to mach, other registers will be excluded
                                if not (reg == mref[1] or reg == mref[2]):
                                    # reg can not be it
                                    possible_idx_regs.remove(reg)

                    idx_reg = None
                    if len(possible_idx_regs) == 1:
                        # good, exactly one register was found
                        idx_reg = possible_idx_regs[0]
                    elif possible_idx_regs and itemsEqual([increments[pidxreg]
                                                           for pidxreg in possible_idx_regs]):
                        # multiple were option found, but all have the same increment
                        # use first match:
                        idx_reg = possible_idx_regs[0]

                    if idx_reg:
                        mem_scales = [mref[3] for mref in refs
                                      if idx_reg == mref[2] or idx_reg == mref[1]]

                        if itemsEqual(mem_scales):
                            # good, all scales are equal
                            try:
                                pointer_increment = mem_scales[0] * increments[idx_reg]
                            except:
                                print("labels", pformat(labels[labels.index(last_label):]))
                                print("lines", pformat(asm_lines[last_label_line:i + 1]))
                                print("increments", increments)
                                print("mem_references", pformat(mem_references))
                                print("idx_reg", idx_reg)
                                print("mem_scales", mem_scales)
                                raise

                blocks.append({'first_line': last_label_line,
                               'last_line': i,
                               'ops': i - last_label_line,
                               'labels': labels[labels.index(last_label):],
                               'packed_instr': packed_ctr,
                               'avx_instr': avx_ctr,
                               'XMM': (len(xmm_references), len(set(xmm_references))),
                               'YMM': (len(ymm_references), len(set(ymm_references))),
                               'ZMM': (len(zmm_references), len(set(zmm_references))),
                               'GP': (len(gp_references), len(set(gp_references))),
                               'regs': (len(xmm_references) + len(ymm_references) +
                                        len(zmm_references) + len(gp_references),
                                        len(set(xmm_references)) + len(set(ymm_references)) +
                                        len(set(zmm_references)) +
                                        len(set(gp_references))),
                               'pointer_increment': pointer_increment,
                               'lines': asm_lines[last_label_line:i + 1],
                               'possible_idx_regs': possible_idx_regs,
                               'mem_references': mem_references,
                               'increments': increments, })
            # Reset counters
            packed_ctr = 0
            avx_ctr = 0
            xmm_references = []
            ymm_references = []
            zmm_references = []
            gp_references = []
            mem_references = []
            increments = {}
            last_labels = OrderedDict()
    return list(enumerate(blocks))


def select_best_block(blocks):
    """Return best block selected based on simple heuristic."""
    # TODO make this cleverer with more stats
    if not blocks:
        raise ValueError("No suitable blocks were found in assembly.")
    best_block = max(blocks, key=lambda b: b[1]['packed_instr'])
    if best_block[1]['packed_instr'] == 0:
        best_block = max(blocks,
                         key=lambda b: (b[1]['ops'] + b[1]['packed_instr'] + b[1]['avx_instr'],
                                        b[1]['ZMM'], b[1]['YMM'], b[1]['XMM']))
    return best_block[0]


def userselect_increment(block):
    """Let user interactively select byte increment."""
    print("Selected block:")
    print('\n    ' + ('\n    '.join(block['lines'])))
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


def userselect_block(blocks, default=None, debug=False):
    """Let user interactively select block."""
    print("Blocks found in assembly file:")
    print("      block     | OPs | pck. | AVX || Registers |    ZMM   |    YMM   |    XMM   |"
          "GP   ||ptr.inc|\n"
          "----------------+-----+------+-----++-----------+----------+----------+----------+"
          "---------++-------|")
    for idx, b in blocks:
        print('{:>2} {b[labels]!r:>12} | {b[ops]:>3} | {b[packed_instr]:>4} | {b[avx_instr]:>3} |'
              '| {b[regs][0]:>3} ({b[regs][1]:>3}) | {b[ZMM][0]:>3} ({b[ZMM][1]:>2}) | '
              '{b[YMM][0]:>3} ({b[YMM][1]:>2}) | '
              '{b[XMM][0]:>3} ({b[XMM][1]:>2}) | {b[GP][0]:>2} ({b[GP][1]:>2}) || '
              '{b[pointer_increment]!s:>5} |'.format(idx, b=b))

        if debug:
            ln = b['first_line']
            print(' '*4 + 'Code:')
            for l in b['lines']:
                print(' '*8 + '{:>5} | {}'.format(ln, l))
                ln += 1
            print(' '*4 + 'Metadata:')
            print(textwrap.indent(
                pformat({k: v for k,v in b.items() if k not in ['lines']}),
                ' '*8))

    # Let user select block:
    block_idx = -1
    while not (0 <= block_idx < len(blocks)):
        block_idx = input("Choose block to be marked [" + str(default) + "]: ") or default
        try:
            block_idx = int(block_idx)
        except ValueError:
            block_idx = -1
    # block = blocks[block_idx][1]

    return block_idx


def insert_markers(asm_lines, start_line, end_line):
    """Insert IACA marker into list of ASM instructions at given indices."""
    asm_lines = (asm_lines[:start_line] + START_MARKER +
                 asm_lines[start_line:end_line + 1] + END_MARKER +
                 asm_lines[end_line + 1:])
    return asm_lines


def iaca_instrumentation(input_file, output_file,
                         block_selection='auto',
                         pointer_increment='auto_with_manual_fallback',
                         debug=False):
    """
    Add IACA markers to an assembly file.

    If instrumentation fails because loop increment could not determined automatically, a ValueError
    is raised.

    :param input_file: file-like object to read from
    :param output_file: file-like object to write to
    :param block_selection: index of the assembly block to instrument, or 'auto' for automatically
                            using block with the
                            most vector instructions, or 'manual' to read index to prompt user
    :param pointer_increment: number of bytes the pointer is incremented after the loop or
                              - 'auto': automatic detection, otherwise RuntimeError is raised
                              - 'auto_with_manual_fallback': like auto with fallback to manual input
                              - 'manual': prompt user
    :param debug: output additional internal analysis information. Only works with manual selection.
    :return: the instrumented assembly block
    """
    assembly_orig = input_file.readlines()

    # If input and output files are the same, overwrite with output
    if input_file is output_file:
        output_file.seek(0)
        output_file.truncate()

    if debug:
        block_selection = 'manual'

    assembly = strip_and_uncomment(copy(assembly_orig))
    assembly = strip_unreferenced_labels(assembly)
    blocks = find_asm_blocks(assembly)
    if block_selection == 'auto':
        block_idx = select_best_block(blocks)
    elif block_selection == 'manual':
        block_idx = userselect_block(blocks, default=select_best_block(blocks), debug=debug)
    elif isinstance(block_selection, int):
        block_idx = block_selection
    else:
        raise ValueError("block_selection has to be an integer, 'auto' or 'manual' ")

    block = blocks[block_idx][1]

    if pointer_increment == 'auto':
        if block['pointer_increment'] is None:
            raise RuntimeError("pointer_increment could not be detected automatically. Use "
                               "--pointer-increment to set manually to byte offset of store "
                               "pointer address between consecutive assembly block iterations.")
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

    instrumented_asm = insert_markers(assembly_orig, block['first_line'], block['last_line'])
    output_file.writelines(instrumented_asm)

    return block


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
    with open(instrumented_assembly_file) as f:
        osaca = OSACA(micro_architecture, f.read())
    result['output'] = osaca.create_output()
    result['port cycles'] = OrderedDict(osaca.get_port_occupation_cycles())
    result['throughput'] = osaca.get_total_throughput()
    result['uops'] = None  # Not given by OSACA

    unmatched_ratio = osaca.get_unmatched_instruction_ratio()
    if unmatched_ratio > 0.1:
        print('WARNING: {:.0%} of the instruction could not be matched during incore analysis '
              'with OSACA. Fix this by extending OSACAs instruction form database with the '
              'required instructions.'.format(unmatched_ratio),
              file=sys.stderr)

    return result


def llvm_mca_analyse_instrumented_assembly(instrumented_assembly_file, micro_architecture):
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
        assembly_section = extract_marked_section(f.read())

    output = subprocess.check_output(['llvm-mca']+micro_architecture.split(' '),
                                     input=assembly_section.encode('utf-8')).decode('utf-8')
    result['output'] = output

    # Extract port names
    port_names = OrderedDict()
    m = re.search(r'Resources:\n(?:[^\n]+\n)+', output)
    for m in re.finditer(r'(\[[0-9]+\])\s+-\s+([a-zA-Z0-9]+)', m.group()):
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
    uops_raw = re.search(r'\n\[1\](\s+\[[0-9]+\]\s+)+Instructions:\n(:?\s*[0-9]+\s+[^\n]+\n)+',
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
                        help='Output nternal analysis information for debugging.')
    args = parser.parse_args()

    # pointer_increment is given, since it makes no difference on the command lien and requires
    # less user input
    iaca_instrumentation(input_file=args.source, output_file=args.outfile,
                         block_selection='manual', pointer_increment=1, debug=args.debug)


if __name__ == '__main__':
    main()
