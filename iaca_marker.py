#!/usr/bin/env python

from __future__ import print_function

import re
import sys

START_MARKER = ['        movl      $111, %ebx # INSERTED BY IACA MARKER UTILITY\n'
                '        .byte     100        # INSERTED BY IACA MARKER UTILITY\n'
                '        .byte     103        # INSERTED BY IACA MARKER UTILITY\n'
                '        .byte     144        # INSERTED BY IACA MARKER UTILITY\n']
END_MARKER = ['        movl      $222, %ebx # INSERTED BY IACA MARKER UTILITY\n'
              '        .byte     100        # INSERTED BY IACA MARKER UTILITY\n'
              '        .byte     103        # INSERTED BY IACA MARKER UTILITY\n'
              '        .byte     144        # INSERTED BY IACA MARKER UTILITY\n']

def find_codeblocks(code_lines):
    blocks = []
    
    last_label_line = -1
    last_label = None
    packed_ctr = 0
    for i, line in enumerate(lines):
        if re.match(r"^v(mul|add|sub|div)p[ds].*", line.strip()):
            packed_ctr += 1
        # TODO add more statistics: half-width ops and mov
        elif line.startswith('..'):
           last_label = line[0:line.find(':')]
           last_label_line = i
        elif '# Prob' in line:
            blocks.append({'first_line': last_label_line,
                           'last_line': i,
                           'lines': i-last_label_line,
                           'label': last_label,
                           'packed_instr': packed_ctr})
            packed_ctr = 0
    
    return blocks

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage:", sys.argv[0], "filename.s")
        sys.exit(1)

    with open(sys.argv[1], 'r') as fp:
        lines = fp.readlines()
    blocks = list(enumerate(find_codeblocks(lines)))
    
    # TODO check for already present markers
    
    print("Blocks found in assembly file:")
    for idx, b in blocks:
        print(str(idx)+": Block", b['label'], "from line", b['first_line'], "to", b['last_line'],
              "(length of", b['lines'], "lines and", b['packed_instr'], "packed instr.)")
    
    # Choose best destfault block:
    # TODO make this cleverer with more stats
    best = max(blocks, key=lambda b: b[1]['packed_instr'])
    if best[1]['packed_instr'] == 0:
        best = max(blocks, key=lambda b: b[1]['lines'])
    
    # Let user select block:
    block_idx = -1
    while 0 >= block_idx < len(blocks):
        block_idx = raw_input("Choose block to be marked ["+str(best[0])+"]: ") or best[0]
        try:
            block_idx = int(block_idx)
        except ValueError:
            block_idx = -1
    block = blocks[block_idx][1]
    
    # Insert markers:
    lines = lines[:block['first_line']] + START_MARKER + \
            lines[block['first_line']:block['last_line']+1] + END_MARKER + \
            lines[block['last_line']+1:]
    with open(sys.argv[1], 'w') as fp:
        fp.writelines(lines)
    
    print("Markers inserted.")
