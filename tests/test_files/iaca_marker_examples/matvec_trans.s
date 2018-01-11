	.section	__TEXT,__text,regular,pure_instructions
	.macosx_version_min 10, 11
	.globl	_main
	.align	4, 0x90
_main:                                  ## @main
	.cfi_startproc
## BB#0:
	pushq	%rbp
Ltmp0:
	.cfi_def_cfa_offset 16
Ltmp1:
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
Ltmp2:
	.cfi_def_cfa_register %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$152, %rsp
Ltmp3:
	.cfi_offset %rbx, -56
Ltmp4:
	.cfi_offset %r12, -48
Ltmp5:
	.cfi_offset %r13, -40
Ltmp6:
	.cfi_offset %r14, -32
Ltmp7:
	.cfi_offset %r15, -24
	movq	%rsi, %rbx
	movq	16(%rbx), %rdi
	callq	_atoi
	movl	%eax, %r14d
	movl	%r14d, -60(%rbp)        ## 4-byte Spill
	movq	8(%rbx), %rdi
	callq	_atoi
                                        ## kill: EAX<def> EAX<kill> RAX<def>
	movl	%eax, %ebx
	movq	%rax, %r12
	imull	%r14d, %ebx
	movslq	%ebx, %rdx
	shlq	$3, %rdx
	leaq	-48(%rbp), %rdi
	movl	$32, %esi
	callq	_posix_memalign
	testl	%eax, %eax
	je	LBB0_1
## BB#2:
	movq	$0, -48(%rbp)
	xorl	%eax, %eax
	jmp	LBB0_3
LBB0_1:                                 ## %._crit_edge.i
	movq	-48(%rbp), %rax
LBB0_3:                                 ## %aligned_malloc.exit
	movq	%rax, -56(%rbp)         ## 8-byte Spill
	testl	%ebx, %ebx
	jle	LBB0_5
## BB#4:                                ## %.lr.ph28
	decl	%ebx
	leaq	8(,%rbx,8), %rdx
	leaq	_.memset_pattern2(%rip), %rsi
	movq	-56(%rbp), %rdi         ## 8-byte Reload
	callq	_memset_pattern16
LBB0_5:
	movq	_var_false@GOTPCREL(%rip), %r13
	cmpl	$0, (%r13)
	je	LBB0_7
## BB#6:
	movq	-56(%rbp), %rdi         ## 8-byte Reload
	callq	_dummy
LBB0_7:
	movq	%r12, %rbx
	movslq	%ebx, %r14
	leaq	(,%r14,8), %rdx
	movq	%rdx, -80(%rbp)         ## 8-byte Spill
	leaq	-48(%rbp), %rdi
	movl	$32, %esi
	callq	_posix_memalign
	testl	%eax, %eax
	je	LBB0_8
## BB#9:
	movq	$0, -48(%rbp)
	xorl	%r12d, %r12d
	jmp	LBB0_10
LBB0_8:                                 ## %._crit_edge.i6
	movq	-48(%rbp), %r12
LBB0_10:                                ## %aligned_malloc.exit7
	testl	%ebx, %ebx
	jle	LBB0_12
## BB#11:                               ## %.lr.ph25
	leal	-1(%rbx), %eax
	leaq	8(,%rax,8), %rdx
	leaq	_.memset_pattern2(%rip), %rsi
	movq	%r12, %rdi
	callq	_memset_pattern16
LBB0_12:
	cmpl	$0, (%r13)
	je	LBB0_14
## BB#13:
	movq	%r12, %rdi
	callq	_dummy
LBB0_14:
	leaq	-48(%rbp), %rdi
	movl	$32, %esi
	movq	-80(%rbp), %rdx         ## 8-byte Reload
	callq	_posix_memalign
	testl	%eax, %eax
	je	LBB0_15
## BB#16:
	movq	$0, -48(%rbp)
	xorl	%r15d, %r15d
	jmp	LBB0_17
LBB0_15:                                ## %._crit_edge.i10
	movq	-48(%rbp), %r15
LBB0_17:                                ## %aligned_malloc.exit11
	testl	%ebx, %ebx
	jle	LBB0_19
## BB#18:                               ## %.lr.ph22
	leal	-1(%rbx), %eax
	leaq	8(,%rax,8), %rdx
	leaq	_.memset_pattern2(%rip), %rsi
	movq	%r15, %rdi
	callq	_memset_pattern16
LBB0_19:
	cmpl	$0, (%r13)
	je	LBB0_21
## BB#20:
	movq	%r15, %rdi
	callq	_dummy
LBB0_21:                                ## %.preheader17
	testl	%ebx, %ebx
	setle	%al
	cmpl	$0, -60(%rbp)           ## 4-byte Folded Reload
	movq	%r14, %r13
	movq	%r13, -104(%rbp)        ## 8-byte Spill
	jle	LBB0_37
## BB#22:                               ## %.preheader17
	testb	%al, %al
	jne	LBB0_37
## BB#23:                               ## %.preheader.lr.ph.split.us
	movq	%rbx, -136(%rbp)        ## 8-byte Spill
	leal	-1(%rbx), %eax
	movq	%rax, -72(%rbp)         ## 8-byte Spill
	decl	-60(%rbp)               ## 4-byte Folded Spill
	leaq	1(%rax), %rdx
	movq	%rdx, -128(%rbp)        ## 8-byte Spill
	movq	%r15, -96(%rbp)         ## 8-byte Spill
	leaq	(%r15,%rax,8), %rcx
	movq	%rcx, -112(%rbp)        ## 8-byte Spill
	movq	%r12, -88(%rbp)         ## 8-byte Spill
	leaq	(%r12,%rax,8), %rax
	movq	%rax, -120(%rbp)        ## 8-byte Spill
	movq	-56(%rbp), %rax         ## 8-byte Reload
	leaq	32(%rax), %r14
	leaq	32(%r15), %rax
	movq	%rax, -144(%rbp)        ## 8-byte Spill
	leaq	32(%r12), %rax
	movq	%rax, -152(%rbp)        ## 8-byte Spill
	movq	%rdx, %rax
	andq	$-8, %rax
	movq	%rax, -160(%rbp)        ## 8-byte Spill
	leaq	8(%r12), %rax
	movq	%rax, -168(%rbp)        ## 8-byte Spill
	leaq	8(%r15), %rax
	movq	%rax, -176(%rbp)        ## 8-byte Spill
	leal	1(%rbx), %eax
	movl	%eax, -180(%rbp)        ## 4-byte Spill
	xorl	%r8d, %r8d
	movl	$1, %r11d
	.align	4, 0x90
LBB0_26:                                ## %.lr.ph.us
                                        ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_30 Depth 2
                                        ##     Child Loop BB0_24 Depth 2
	movq	%r13, %r15
	imulq	%r8, %r15
	movq	%rdx, %r10
	movl	$0, %r9d
	movabsq	$8589934584, %rax       ## imm = 0x1FFFFFFF8
	andq	%rax, %r10
	je	LBB0_32
## BB#27:                               ## %vector.memcheck
                                        ##   in Loop: Header=BB0_26 Depth=1
	movq	-72(%rbp), %rax         ## 8-byte Reload
	leaq	(%rax,%r15), %rdx
	movq	-56(%rbp), %rax         ## 8-byte Reload
	leaq	(%rax,%rdx,8), %rdx
	leaq	(%rax,%r15,8), %rsi
	movq	-96(%rbp), %rdi         ## 8-byte Reload
	cmpq	%rdx, %rdi
	setbe	%bl
	movq	-112(%rbp), %rcx        ## 8-byte Reload
	cmpq	%rcx, %rsi
	setbe	%al
	cmpq	-120(%rbp), %rdi        ## 8-byte Folded Reload
	setbe	%dl
	cmpq	%rcx, -88(%rbp)         ## 8-byte Folded Reload
	setbe	%r13b
	testb	%al, %bl
	jne	LBB0_28
## BB#29:                               ## %vector.memcheck
                                        ##   in Loop: Header=BB0_26 Depth=1
	movq	-160(%rbp), %rsi        ## 8-byte Reload
	movq	-152(%rbp), %r12        ## 8-byte Reload
	movq	-144(%rbp), %rbx        ## 8-byte Reload
	movq	%r14, %rdi
	movl	$0, %r9d
	andb	%r13b, %dl
	movq	-104(%rbp), %r13        ## 8-byte Reload
	movq	-128(%rbp), %rdx        ## 8-byte Reload
	jne	LBB0_32
	.align	4, 0x90
LBB0_30:                                ## %vector.body
                                        ##   Parent Loop BB0_26 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	vmovupd	-32(%rdi), %xmm0
	vinsertf128	$1, -16(%rdi), %ymm0, %ymm0
	vmovupd	(%rdi), %xmm1
	vinsertf128	$1, 16(%rdi), %ymm1, %ymm1
	vmovupd	-32(%r12), %xmm2
	vinsertf128	$1, -16(%r12), %ymm2, %ymm2
	vmovupd	(%r12), %xmm3
	vinsertf128	$1, 16(%r12), %ymm3, %ymm3
	vmulpd	%ymm2, %ymm0, %ymm0
	vmulpd	%ymm3, %ymm1, %ymm1
	vmovupd	-32(%rbx), %xmm2
	vinsertf128	$1, -16(%rbx), %ymm2, %ymm2
	vmovupd	(%rbx), %xmm3
	vinsertf128	$1, 16(%rbx), %ymm3, %ymm3
	vaddpd	%ymm0, %ymm2, %ymm0
	vaddpd	%ymm1, %ymm3, %ymm1
	vextractf128	$1, %ymm0, -16(%rbx)
	vmovupd	%xmm0, -32(%rbx)
	vextractf128	$1, %ymm1, 16(%rbx)
	vmovupd	%xmm1, (%rbx)
	addq	$64, %rdi
	addq	$64, %rbx
	addq	$64, %r12
	addq	$-8, %rsi
	jne	LBB0_30
## BB#31:                               ##   in Loop: Header=BB0_26 Depth=1
	movq	%r10, %r9
	jmp	LBB0_32
LBB0_28:                                ##   in Loop: Header=BB0_26 Depth=1
	xorl	%r9d, %r9d
	movq	-104(%rbp), %r13        ## 8-byte Reload
	movq	-128(%rbp), %rdx        ## 8-byte Reload
	.align	4, 0x90
LBB0_32:                                ## %middle.block
                                        ##   in Loop: Header=BB0_26 Depth=1
	cmpq	%r9, %rdx
	je	LBB0_25
## BB#33:                               ## %scalar.ph.preheader
                                        ##   in Loop: Header=BB0_26 Depth=1
	movq	-136(%rbp), %rax        ## 8-byte Reload
                                        ## kill: EAX<def> EAX<kill> RAX<kill>
	subl	%r9d, %eax
	movq	-72(%rbp), %rcx         ## 8-byte Reload
                                        ## kill: ECX<def> ECX<kill> RCX<kill>
	subl	%r9d, %ecx
	testb	$1, %al
	je	LBB0_35
## BB#34:                               ## %scalar.ph.prol
                                        ##   in Loop: Header=BB0_26 Depth=1
	addq	%r9, %r15
	movq	-56(%rbp), %rax         ## 8-byte Reload
	vmovsd	(%rax,%r15,8), %xmm0    ## xmm0 = mem[0],zero
	movq	-88(%rbp), %rax         ## 8-byte Reload
	vmulsd	(%rax,%r9,8), %xmm0, %xmm0
	movq	-96(%rbp), %rax         ## 8-byte Reload
	vaddsd	(%rax,%r9,8), %xmm0, %xmm0
	vmovsd	%xmm0, (%rax,%r9,8)
	incq	%r9
LBB0_35:                                ## %scalar.ph.preheader.split
                                        ##   in Loop: Header=BB0_26 Depth=1
	testl	%ecx, %ecx
	je	LBB0_25
## BB#36:                               ## %scalar.ph.preheader.split.split
                                        ##   in Loop: Header=BB0_26 Depth=1
	movq	-168(%rbp), %rax        ## 8-byte Reload
	leaq	(%rax,%r9,8), %rcx
	movq	-176(%rbp), %rax        ## 8-byte Reload
	leaq	(%rax,%r9,8), %rsi
	leaq	(%r9,%r11), %rax
	movq	-56(%rbp), %rdi         ## 8-byte Reload
	leaq	(%rdi,%rax,8), %rdi
	leal	1(%r9), %eax
	movl	-180(%rbp), %ebx        ## 4-byte Reload
	subl	%eax, %ebx
	.align	4, 0x90
LBB0_24:                                ## %scalar.ph
                                        ##   Parent Loop BB0_26 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	vmovsd	-8(%rdi), %xmm0         ## xmm0 = mem[0],zero
	vmulsd	-8(%rcx), %xmm0, %xmm0
	vaddsd	-8(%rsi), %xmm0, %xmm0
	vmovsd	%xmm0, -8(%rsi)
	vmovsd	(%rdi), %xmm0           ## xmm0 = mem[0],zero
	vmulsd	(%rcx), %xmm0, %xmm0
	vaddsd	(%rsi), %xmm0, %xmm0
	vmovsd	%xmm0, (%rsi)
	addq	$2, %r9
	addq	$16, %rcx
	addq	$16, %rsi
	addq	$16, %rdi
	addl	$-2, %ebx
	jne	LBB0_24
LBB0_25:                                ##   in Loop: Header=BB0_26 Depth=1
	leaq	1(%r8), %rcx
	addq	-80(%rbp), %r14         ## 8-byte Folded Reload
	addq	%r13, %r11
	cmpl	-60(%rbp), %r8d         ## 4-byte Folded Reload
	movq	%rcx, %r8
	jne	LBB0_26
LBB0_37:                                ## %._crit_edge20
	xorl	%eax, %eax
	addq	$152, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	vzeroupper
	retq
	.cfi_endproc

	.section	__TEXT,__const
	.align	4                       ## @.memset_pattern2
_.memset_pattern2:
	.quad	4597454643604897137     ## double 2.300000e-01
	.quad	4597454643604897137     ## double 2.300000e-01


.subsections_via_symbols
