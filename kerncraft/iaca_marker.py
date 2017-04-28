#!/usr/bin/env python

from __future__ import print_function
from __future__ import absolute_import

# Version check
import sys
if sys.version_info[0] == 2 and sys.version_info < (2, 7) or \
        sys.version_info[0] == 3 and sys.version_info < (3, 4):
    print("Must use python 2.7 or 3.4 and greater.", file=sys.stderr)
    sys.exit(1)

import re
from six.moves import map
from six.moves import input

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


def find_asm_blocks(asm_lines):
    '''
    finds blocks probably corresponding to loops in assembly
    '''
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

        # Strip comments and whitespaces
        line = line.split('#')[0]
        line = line.strip()

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
        elif re.search(r'\d*\(%\w+,%\w+(,\d)?\)$', line):
            m = re.search(r'(?P<off>\d*)\(%(?P<basep>\w+),%(?P<idx>\w+)(?:,(?P<scale>\d))?\)$',
                          line)
            mem_references.append((
                int(m.group('off')) if m.group('off') else 0,
                m.group('basep'),
                m.group('idx'),
                int(m.group('scale')) if m.group('scale') else 1))
        elif last_label and re.match(r'^j[a-z]+\s+'+re.escape(last_label)+r'\s*', line):
            # End of block
            # deduce loop increment from memory index register
            pointer_increment = None  # default -> can not decide, let user choose
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

                    mem_scales = [mref[3] for mref in mem_references if idx_reg == mref[2]]

                    if mem_scales[1:] == mem_scales[:-1]:
                        # good, all scales are equal
                        pointer_increment = mem_scales[0]*increments[idx_reg]

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
                           'lines': asm_lines[last_label_line:i+1],})

    return list(enumerate(blocks))


def select_best_block(blocks):
    # TODO make this cleverer with more stats
    best_block = max(blocks, key=lambda b: b[1]['packed_instr'])
    if best_block[1]['packed_instr'] == 0:
        best_block = max(blocks, key=lambda b: b[1]['ops']+b[1]['packed_instr']+b[1]['avx_instr'])

    return best_block[0]


def userselect_increment(block):
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
    print("Blocks found in assembly file:")
    print("   block   | OPs | pck. | AVX || Registers |    YMM   |    XMM   |    GP   ||ptr.inc|\n"
          "-----------+-----+------+-----++-----------+----------+----------+---------++-------|")
    from pprint import pprint
    #pprint(blocks)
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
    asm_lines = asm_lines[:start_line] + START_MARKER + \
        asm_lines[start_line:end_line+1] + END_MARKER + \
        asm_lines[end_line+1:]
    return asm_lines


def main():
    if len(sys.argv) != 2:
        print("Usage:", sys.argv[0], "filename.s")
        sys.exit(1)

    with open(sys.argv[1], 'r') as fp:
        lines = fp.readlines()
    lines = list(map(str.strip, lines))
    blocks = find_asm_blocks(lines)

    # TODO check for already present markers

    # Choose best default block:
    best_idx = select_best_block(blocks)
    # Let user select block:
    block_idx = userselect_block(blocks, best_idx)

    block = blocks[block_idx][1]

    # Insert markers:
    lines = insert_markers(lines, block['first_line'], block['last_line'])

    # write back to file
    with open(sys.argv[1], 'w') as fp:
        fp.write('\n'.join(lines))

    print("Markers inserted.")

if __name__ == '__main__':
    main()
