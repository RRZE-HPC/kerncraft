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

def find_asm_blocks(asm_lines):
    blocks = []
    
    last_label_line = -1
    last_label = None
    packed_ctr = 0
    for i, line in enumerate(asm_lines):
        if re.match(r"^[v]?(mul|add|sub|div)[h]?p[ds]", line.strip()):
            packed_ctr += 1
        # TODO add more statistics: half-width ops and mov
        elif re.match(r'^..B[0-9]+\.[0-9]+:', line):
           last_label = line[0:line.find(':')]
           last_label_line = i
        elif '# Prob' in line:
            blocks.append({'first_line': last_label_line,
                           'last_line': i,
                           'lines': i-last_label_line,
                           'label': last_label,
                           'packed_instr': packed_ctr})
            packed_ctr = 0
    
    return list(enumerate(blocks))

def select_best_block(blocks):
    # TODO make this cleverer with more stats
    best_block = max(blocks, key=lambda b: b[1]['packed_instr'])
    if best_block[1]['packed_instr'] == 0:
        best_block = max(blocks, key=lambda b: b[1]['lines'])
    
    return best_block[0]
    
def userselect_block(blocks, default=None):
    print("Blocks found in assembly file:")
    for idx, b in blocks:
        print(str(idx)+": Block", b['label'], "from line", b['first_line'], "to", b['last_line'],
              "(length of", b['lines'], "lines and", b['packed_instr'], "packed instr.)")
    
    # Let user select block:
    block_idx = -1
    while 0 >= block_idx < len(blocks):
        block_idx = raw_input("Choose block to be marked ["+str(default)+"]: ") or default
        try:
            block_idx = int(block_idx)
        except ValueError:
            block_idx = -1
    block = blocks[block_idx][1]
    
    return block_idx

def insert_markers(asm_lines, start_line, end_line):
    asm_lines = asm_lines[:start_line] + START_MARKER + \
                asm_lines[start_line:end_line+1] + END_MARKER + \
                asm_lines[end_line+1:]
    return asm_lines

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage:", sys.argv[0], "filename.s")
        sys.exit(1)

    with open(sys.argv[1], 'r') as fp:
        lines = fp.readlines()
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
        fp.writelines(lines)
    
    print("Markers inserted.")
