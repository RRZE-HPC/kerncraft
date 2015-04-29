# mark_description "Intel(R) C Intel(R) 64 Compiler XE for applications running on Intel(R) 64, Version 14.0.0.074 Build 2013071";
# mark_description "6";
# mark_description "-std=c99 -S -xHost -O3 -fno-alias -restrict";
	.file "dummy.c"
	.section	__TEXT, __text
L_TXTST0:
# -- Begin  _dummy
# mark_begin;
       .align    4
	.globl _dummy
_dummy:
# parameter 1: %rdi
L_B1.1:                         # Preds L_B1.0
L____tag_value__dummy.1:                                        #1.23
        ret                                                     #1.24
        .align    4
L____tag_value__dummy.3:                                        #
                                # LOE
# mark_end;
	.section	__DATA, __data
# -- End  _dummy
	.section	__DATA, __data
	.globl _dummy.eh
// -- Begin SEGMENT __eh_frame
	.section __TEXT,__eh_frame,coalesced,no_toc+strip_static_syms+live_support
__eh_frame_seg:
L.__eh_frame_seg:
EH_frame0:
L_fde_cie_0:
	.long 0x00000014
	.long 0x00000000
	.long 0x00527a01
	.long 0x01107801
	.long 0x08070c10
	.long 0x01900190
_dummy.eh:
	.long 0x0000001c
	.long _dummy.eh-L_fde_cie_0+0x4
	.quad L____tag_value__dummy.1-_dummy.eh-0x8
	.set L_Qlab1,L____tag_value__dummy.3-L____tag_value__dummy.1
	.quad L_Qlab1
	.long 0x00000000
	.long 0x00000000
# End
	.subsections_via_symbols
