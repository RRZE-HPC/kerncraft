#!/usr/bin/env python

from __future__ import print_function

import re
import sys

START_MARKER = ['        movl      $111, %ebx # INSERTED BY KERNCRAFT IACA MARKER UTILITY\n'
                '        .byte     100        # INSERTED BY KERNCRAFT IACA MARKER UTILITY\n'
                '        .byte     103        # INSERTED BY KERNCRAFT IACA MARKER UTILITY\n'
                '        .byte     144        # INSERTED BY KERNCRAFT IACA MARKER UTILITY\n']
END_MARKER = ['        movl      $222, %ebx # INSERTED BY KERNCRAFT IACA MARKER UTILITY\n'
              '        .byte     100        # INSERTED BY KERNCRAFT IACA MARKER UTILITY\n'
              '        .byte     103        # INSERTED BY KERNCRAFT IACA MARKER UTILITY\n'
              '        .byte     144        # INSERTED BY KERNCRAFT IACA MARKER UTILITY\n']

def find_asm_blocks(asm_lines, with_nop=True):
    '''
    if with_nop is True, only blocks within nop markers will be considered
    '''
    blocks = []
    
    last_label_line = -1
    last_label = None
    packed_ctr = 0
    avx_ctr = 0
    xmm_references = []
    ymm_references = []
    gp_references = []
    last_incr = None
    within_nop_region = False
    for i, line in enumerate(asm_lines):
        if with_nop:
            if line.strip().startswith("nop"):
                within_nop_region = not within_nop_region
            if not within_nop_region:
                continue
        
        # Register access counts
        ymm_references += re.findall('%ymm[0-9]+', line)
        xmm_references += re.findall('%xmm[0-9]+', line)
        gp_references += re.findall('%r[a-z0-9]+', line)
        
        if re.match(r"^[v]?(mul|add|sub|div)[h]?p[ds]", line.strip()):
            if line.strip().startswith('v'):
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
            last_incr = None
        elif re.match(r'^inc[bwlq]?', line.strip()):
            last_incr = 1
        elif re.match(r'^add[bwlq]?\s+\$[0-9]+,', line.strip()):
            const_start = line.find('$')+1
            const_end = line[const_start+1:].find(',')+const_start+1
            last_incr = int(line[const_start:const_end])
        elif re.match(r'^dec[bwlq]?', line.strip()):
            last_incr = -1
        elif re.match(r'^sub[bwlq]?\s+\$[0-9]+,', line.strip()):
            const_start = line.find('$')+1
            const_end = line[const_start+1:].find(',')+const_start+1
            last_incr = -int(line[const_start:const_end])
        elif last_label and re.match(r'^j[a-z]+\s+'+re.escape(last_label)+'\s+', line.strip()):
            blocks.append({'first_line': last_label_line,
                           'last_line': i,
                           'lines': i-last_label_line,
                           'label': last_label,
                           'packed_instr': packed_ctr,
                           'avx_instr': avx_ctr,
                           'XMM': (len(xmm_references), len(set(xmm_references))),
                           'YMM': (len(ymm_references), len(set(ymm_references))),
                           'GP': (len(gp_references), len(set(gp_references))),
                           'regs': (len(xmm_references) + len(ymm_references) + len(gp_references),
                                    len(set(xmm_references)) + len(set(ymm_references)) + 
                                        len(set(gp_references))),
                           'loop_increment': last_incr, })
    
    return list(enumerate(blocks))

def select_best_block(blocks):
    # TODO make this cleverer with more stats
    best_block = max(blocks, key=lambda b: b[1]['packed_instr'])
    if best_block[1]['packed_instr'] == 0:
        best_block = max(blocks, key=lambda b: b[1]['lines'])
    
    return best_block[0]
    
def userselect_block(blocks, default=None):
    print("Blocks found in assembly file:")
    print("   block   | OPs | pck. | AVX || Registers |    YMM   |    XMM   |    GP   || l.inc |\n"+ 
          "-----------+-----+------+-----++-----------+----------+----------+---------++-------|")
    for idx, b in blocks:
        print(
            '{:>2} {b[label]:>5} | {b[lines]:>3} | {b[packed_instr]:>4} | {b[avx_instr]:>3} |'.format(idx, b=b)+
            '| {b[regs][0]:>3} ({b[regs][1]:>3}) | {b[YMM][0]:>3} ({b[YMM][1]:>2}) | {b[XMM][0]:>3} ({b[XMM][1]:>2}) | {b[GP][0]:>2} ({b[GP][1]:>2}) || {b[loop_increment]:>5} |'.format(b=b))
        
    # Let user select block:
    block_idx = -1
    while not (0 <= block_idx < len(blocks)):
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

def main():
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

if __name__ == '__main__':
    main()
