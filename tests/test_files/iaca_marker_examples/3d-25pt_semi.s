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
	subq	$360, %rsp              ## imm = 0x168
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
	movq	8(%rsi), %rdi
	callq	_atoi
	movl	%eax, %r12d
	movl	%r12d, %ecx
	imull	%ecx, %ecx
	movl	%ecx, -388(%rbp)        ## 4-byte Spill
	movl	%ecx, %ebx
	imull	%r12d, %ebx
	movslq	%ebx, %r15
	shlq	$3, %r15
	leaq	-48(%rbp), %rdi
	movl	$32, %esi
	movq	%r15, %rdx
	callq	_posix_memalign
	testl	%eax, %eax
	je	LBB0_1
## BB#2:
	movq	$0, -48(%rbp)
	xorl	%r14d, %r14d
	jmp	LBB0_3
LBB0_1:                                 ## %._crit_edge.i
	movq	-48(%rbp), %r14
LBB0_3:                                 ## %aligned_malloc.exit
	testl	%ebx, %ebx
	jle	LBB0_5
## BB#4:                                ## %.lr.ph20.preheader
	leal	-1(%rbx), %eax
	leaq	8(,%rax,8), %rdx
	leaq	L_.memset_pattern.1(%rip), %rsi
	movq	%r14, %rdi
	callq	_memset_pattern16
LBB0_5:                                 ## %._crit_edge.21
	movq	%r14, -344(%rbp)        ## 8-byte Spill
	movq	_var_false@GOTPCREL(%rip), %r13
	cmpl	$0, (%r13)
	je	LBB0_7
## BB#6:
	movq	-344(%rbp), %rdi        ## 8-byte Reload
	callq	_dummy
LBB0_7:
	leaq	-48(%rbp), %rdi
	movl	$32, %esi
	movq	%r15, %rdx
	callq	_posix_memalign
	testl	%eax, %eax
	je	LBB0_8
## BB#9:
	movq	$0, -48(%rbp)
	xorl	%edi, %edi
	jmp	LBB0_10
LBB0_8:                                 ## %._crit_edge.i.5
	movq	-48(%rbp), %rdi
LBB0_10:                                ## %aligned_malloc.exit6
	testl	%ebx, %ebx
	jle	LBB0_12
## BB#11:                               ## %.lr.ph17.preheader
	decl	%ebx
	leaq	8(,%rbx,8), %rdx
	leaq	L_.memset_pattern.1(%rip), %rsi
	movq	%rdi, %rbx
	callq	_memset_pattern16
	movq	%rbx, %rdi
LBB0_12:                                ## %._crit_edge.18
	cmpl	$0, (%r13)
	je	LBB0_13
## BB#14:
	movq	%rdi, %rbx
	callq	_dummy
	cmpl	$0, (%r13)
	movabsq	$4597454643604897137, %r15 ## imm = 0x3FCD70A3D70A3D71
	movq	%r15, -48(%rbp)
	je	LBB0_15
## BB#16:
	leaq	-48(%rbp), %rdi
	callq	_dummy
	cmpl	$0, (%r13)
	movq	%r15, -56(%rbp)
	je	LBB0_17
## BB#18:
	leaq	-56(%rbp), %rdi
	callq	_dummy
	cmpl	$0, (%r13)
	movq	%r15, -64(%rbp)
	je	LBB0_19
## BB#20:
	leaq	-64(%rbp), %rdi
	callq	_dummy
	cmpl	$0, (%r13)
	movq	%r15, -72(%rbp)
	je	LBB0_21
## BB#22:
	leaq	-72(%rbp), %rdi
	callq	_dummy
	cmpl	$0, (%r13)
	movq	%r15, -80(%rbp)
	je	LBB0_23
## BB#24:
	leaq	-80(%rbp), %rdi
	callq	_dummy
	cmpl	$0, (%r13)
	movq	%r15, -88(%rbp)
	je	LBB0_25
## BB#26:
	leaq	-88(%rbp), %rdi
	callq	_dummy
	cmpl	$0, (%r13)
	movq	%r15, -96(%rbp)
	je	LBB0_27
## BB#28:
	leaq	-96(%rbp), %rdi
	callq	_dummy
	cmpl	$0, (%r13)
	movq	%r15, -104(%rbp)
	je	LBB0_29
## BB#30:
	leaq	-104(%rbp), %rdi
	callq	_dummy
	cmpl	$0, (%r13)
	movq	%r15, -112(%rbp)
	je	LBB0_31
## BB#32:
	leaq	-112(%rbp), %rdi
	callq	_dummy
	cmpl	$0, (%r13)
	movq	%r15, -120(%rbp)
	je	LBB0_33
## BB#34:
	leaq	-120(%rbp), %rdi
	callq	_dummy
	cmpl	$0, (%r13)
	movq	%r15, -128(%rbp)
	je	LBB0_35
## BB#36:
	leaq	-128(%rbp), %rdi
	callq	_dummy
	cmpl	$0, (%r13)
	movq	%r15, -136(%rbp)
	je	LBB0_37
## BB#38:
	leaq	-136(%rbp), %rdi
	callq	_dummy
	cmpl	$0, (%r13)
	movq	%r15, -144(%rbp)
	je	LBB0_39
## BB#40:
	leaq	-144(%rbp), %rdi
	callq	_dummy
	cmpl	$0, (%r13)
	movq	%r15, -152(%rbp)
	je	LBB0_41
## BB#42:
	leaq	-152(%rbp), %rdi
	callq	_dummy
	cmpl	$0, (%r13)
	movq	%r15, -160(%rbp)
	je	LBB0_43
## BB#44:
	leaq	-160(%rbp), %rdi
	callq	_dummy
	cmpl	$0, (%r13)
	movq	%r15, -168(%rbp)
	je	LBB0_45
## BB#46:
	leaq	-168(%rbp), %rdi
	callq	_dummy
	cmpl	$0, (%r13)
	movq	%r15, -176(%rbp)
	je	LBB0_47
## BB#48:
	leaq	-176(%rbp), %rdi
	callq	_dummy
	cmpl	$0, (%r13)
	movq	%r15, -184(%rbp)
	je	LBB0_49
## BB#50:
	leaq	-184(%rbp), %rdi
	callq	_dummy
	cmpl	$0, (%r13)
	movq	%r15, -192(%rbp)
	je	LBB0_51
## BB#52:
	leaq	-192(%rbp), %rdi
	callq	_dummy
	cmpl	$0, (%r13)
	movq	%r15, -200(%rbp)
	je	LBB0_53
## BB#54:
	leaq	-200(%rbp), %rdi
	callq	_dummy
	cmpl	$0, (%r13)
	movq	%r15, -208(%rbp)
	je	LBB0_55
## BB#56:
	leaq	-208(%rbp), %rdi
	movq	%rbx, -224(%rbp)        ## 8-byte Spill
	callq	_dummy
	cmpl	$0, (%r13)
	movq	%r15, -216(%rbp)
	je	LBB0_58
## BB#57:
	leaq	-216(%rbp), %rdi
	callq	_dummy
	jmp	LBB0_58
LBB0_13:                                ## %.thread
	movq	%rdi, %rbx
	movabsq	$4597454643604897137, %rax ## imm = 0x3FCD70A3D70A3D71
	movq	%rax, -48(%rbp)
LBB0_15:                                ## %.thread63
	movabsq	$4597454643604897137, %rax ## imm = 0x3FCD70A3D70A3D71
	movq	%rax, -56(%rbp)
LBB0_17:                                ## %.thread67
	movabsq	$4597454643604897137, %rax ## imm = 0x3FCD70A3D70A3D71
	movq	%rax, -64(%rbp)
LBB0_19:                                ## %.thread72
	movabsq	$4597454643604897137, %rax ## imm = 0x3FCD70A3D70A3D71
	movq	%rax, -72(%rbp)
LBB0_21:                                ## %.thread78
	movabsq	$4597454643604897137, %rax ## imm = 0x3FCD70A3D70A3D71
	movq	%rax, -80(%rbp)
LBB0_23:                                ## %.thread85
	movabsq	$4597454643604897137, %rax ## imm = 0x3FCD70A3D70A3D71
	movq	%rax, -88(%rbp)
LBB0_25:                                ## %.thread93
	movabsq	$4597454643604897137, %rax ## imm = 0x3FCD70A3D70A3D71
	movq	%rax, -96(%rbp)
LBB0_27:                                ## %.thread102
	movabsq	$4597454643604897137, %rax ## imm = 0x3FCD70A3D70A3D71
	movq	%rax, -104(%rbp)
LBB0_29:                                ## %.thread112
	movabsq	$4597454643604897137, %rax ## imm = 0x3FCD70A3D70A3D71
	movq	%rax, -112(%rbp)
LBB0_31:                                ## %.thread123
	movabsq	$4597454643604897137, %rax ## imm = 0x3FCD70A3D70A3D71
	movq	%rax, -120(%rbp)
LBB0_33:                                ## %.thread135
	movabsq	$4597454643604897137, %rax ## imm = 0x3FCD70A3D70A3D71
	movq	%rax, -128(%rbp)
LBB0_35:                                ## %.thread148
	movabsq	$4597454643604897137, %rax ## imm = 0x3FCD70A3D70A3D71
	movq	%rax, -136(%rbp)
LBB0_37:                                ## %.thread162
	movabsq	$4597454643604897137, %rax ## imm = 0x3FCD70A3D70A3D71
	movq	%rax, -144(%rbp)
LBB0_39:                                ## %.thread177
	movabsq	$4597454643604897137, %rax ## imm = 0x3FCD70A3D70A3D71
	movq	%rax, -152(%rbp)
LBB0_41:                                ## %.thread193
	movabsq	$4597454643604897137, %rax ## imm = 0x3FCD70A3D70A3D71
	movq	%rax, -160(%rbp)
LBB0_43:                                ## %.thread210
	movabsq	$4597454643604897137, %rax ## imm = 0x3FCD70A3D70A3D71
	movq	%rax, -168(%rbp)
LBB0_45:                                ## %.thread228
	movabsq	$4597454643604897137, %rax ## imm = 0x3FCD70A3D70A3D71
	movq	%rax, -176(%rbp)
LBB0_47:                                ## %.thread247
	movabsq	$4597454643604897137, %rax ## imm = 0x3FCD70A3D70A3D71
	movq	%rax, -184(%rbp)
LBB0_49:                                ## %.thread267
	movabsq	$4597454643604897137, %rax ## imm = 0x3FCD70A3D70A3D71
	movq	%rax, -192(%rbp)
LBB0_51:                                ## %.thread288
	movabsq	$4597454643604897137, %rax ## imm = 0x3FCD70A3D70A3D71
	movq	%rax, -200(%rbp)
LBB0_53:                                ## %.thread310
	movabsq	$4597454643604897137, %rax ## imm = 0x3FCD70A3D70A3D71
	movq	%rax, -208(%rbp)
LBB0_55:                                ## %.thread333
	movq	%rbx, -224(%rbp)        ## 8-byte Spill
	movabsq	$4597454643604897137, %rax ## imm = 0x3FCD70A3D70A3D71
	movq	%rax, -216(%rbp)
LBB0_58:                                ## %.preheader8
	leal	-4(%r12), %eax
	movl	%eax, -336(%rbp)        ## 4-byte Spill
	cmpl	$5, %eax
	jl	LBB0_65
## BB#59:                               ## %.preheader7.lr.ph
	movq	%r12, %rsi
	movq	%rsi, -352(%rbp)        ## 8-byte Spill
	leal	4(,%rsi,4), %r13d
	imull	%esi, %r13d
	leal	8(,%rsi,4), %r10d
	imull	%esi, %r10d
	addl	$4, %r10d
	movq	%r10, -280(%rbp)        ## 8-byte Spill
	leal	7(,%rsi,4), %ecx
	imull	%esi, %ecx
	addl	$4, %ecx
	movq	%rcx, -304(%rbp)        ## 8-byte Spill
	leal	6(,%rsi,4), %r15d
	imull	%esi, %r15d
	addl	$4, %r15d
	movq	%r15, -288(%rbp)        ## 8-byte Spill
	leal	5(,%rsi,4), %edx
	imull	%esi, %edx
	addl	$4, %edx
	movq	%rdx, -296(%rbp)        ## 8-byte Spill
	leal	-8(%rsi), %r11d
	leal	4(,%rsi,8), %r8d
	imull	%esi, %r8d
	addl	$4, %r8d
	imull	$7, %esi, %r9d
	addl	$4, %r9d
	imull	%esi, %r9d
	addl	$4, %r9d
	leal	(%rsi,%rsi,2), %eax
	leal	4(%rax,%rax), %r12d
	imull	%esi, %r12d
	addl	$4, %r12d
	leal	4(%rsi,%rsi,4), %ebx
	imull	%esi, %ebx
	addl	$4, %ebx
	leal	4(%r13), %eax
	movl	$4, %esi
	xorl	%edi, %edi
	.align	4, 0x90
LBB0_60:                                ## %.preheader.lr.ph
                                        ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_61 Depth 2
                                        ##       Child Loop BB0_62 Depth 3
	movq	%rdi, -384(%rbp)        ## 8-byte Spill
	movl	%eax, -372(%rbp)        ## 4-byte Spill
	movl	%ebx, -368(%rbp)        ## 4-byte Spill
	movl	%r12d, -364(%rbp)       ## 4-byte Spill
	movl	%r9d, -360(%rbp)        ## 4-byte Spill
	movl	%r8d, -356(%rbp)        ## 4-byte Spill
	incl	%esi
	movl	%esi, -376(%rbp)        ## 4-byte Spill
	movl	%r8d, %esi
	movl	%eax, %r8d
	movl	%r9d, %eax
	movl	%ebx, %r9d
	movl	%eax, %ebx
	movl	%esi, %eax
	movl	%edi, %r14d
	movl	$4, %eax
	.align	4, 0x90
LBB0_61:                                ## %.lr.ph
                                        ##   Parent Loop BB0_60 Depth=1
                                        ## =>  This Loop Header: Depth=2
                                        ##       Child Loop BB0_62 Depth 3
	movl	%eax, -312(%rbp)        ## 4-byte Spill
	movl	%r14d, -308(%rbp)       ## 4-byte Spill
	movl	%esi, -332(%rbp)        ## 4-byte Spill
	movl	%ebx, -328(%rbp)        ## 4-byte Spill
	movl	%r12d, -324(%rbp)       ## 4-byte Spill
	movl	%r9d, -320(%rbp)        ## 4-byte Spill
	movl	%r8d, -316(%rbp)        ## 4-byte Spill
	movslq	%esi, %rax
	movq	-224(%rbp), %r14        ## 8-byte Reload
	leaq	(%r14,%rax,8), %rsi
	movq	%rsi, -232(%rbp)        ## 8-byte Spill
	movq	-344(%rbp), %rdi        ## 8-byte Reload
	leaq	(%rdi,%rax,8), %rax
	movq	%rax, -240(%rbp)        ## 8-byte Spill
	movslq	%ebx, %rax
	movq	%r14, %rbx
	leaq	(%rdi,%rax,8), %rax
	movq	%rax, -248(%rbp)        ## 8-byte Spill
	movslq	%r12d, %rax
	leaq	(%rdi,%rax,8), %rax
	movq	%rax, -256(%rbp)        ## 8-byte Spill
	movslq	%r9d, %rax
	leaq	(%rdi,%rax,8), %rax
	movq	%rax, -264(%rbp)        ## 8-byte Spill
	movslq	%r8d, %rax
	leaq	(%rbx,%rax,8), %r8
	leaq	(%rdi,%rax,8), %rax
	movq	%rax, -272(%rbp)        ## 8-byte Spill
	incl	-312(%rbp)              ## 4-byte Folded Spill
	xorl	%r9d, %r9d
	movl	-308(%rbp), %eax        ## 4-byte Reload
	movl	%eax, %esi
	.align	4, 0x90
LBB0_62:                                ##   Parent Loop BB0_60 Depth=1
                                        ##     Parent Loop BB0_61 Depth=2
                                        ## =>    This Inner Loop Header: Depth=3
	movq	%rbx, -224(%rbp)        ## 8-byte Spill
	movq	%r11, %rbx
	movq	-272(%rbp), %rax        ## 8-byte Reload
	movq	(%rax,%r9,8), %r11
	movq	%r11, -48(%rbp)
	leal	(%rdx,%rsi), %eax
	cltq
	movq	(%rdi,%rax,8), %rax
	movq	%rax, -56(%rbp)
	leal	(%r15,%rsi), %eax
	cltq
	movq	(%rdi,%rax,8), %rax
	movq	%rax, -64(%rbp)
	leal	(%rcx,%rsi), %eax
	cltq
	movq	(%rdi,%rax,8), %rdx
	movq	%rdx, -72(%rbp)
	leal	(%r10,%rsi), %eax
	movslq	%eax, %r14
	movq	(%rdi,%r14,8), %r12
	movq	%r12, -80(%rbp)
	movq	-264(%rbp), %rax        ## 8-byte Reload
	movq	(%rax,%r9,8), %rax
	movq	%rax, -88(%rbp)
	movq	-256(%rbp), %rax        ## 8-byte Reload
	movq	(%rax,%r9,8), %rax
	movq	%rax, -96(%rbp)
	movq	-248(%rbp), %rcx        ## 8-byte Reload
	movq	(%rcx,%r9,8), %r10
	movq	%r10, -104(%rbp)
	movq	-240(%rbp), %r15        ## 8-byte Reload
	vmovsd	(%r15,%r9,8), %xmm12    ## xmm12 = mem[0],zero
	vmovsd	%xmm12, -112(%rbp)
	vmovq	%r11, %xmm8
	movq	%rbx, %r11
	vmovsd	-208(%rbp), %xmm3       ## xmm3 = mem[0],zero
	leal	(%r13,%rsi), %ebx
	movslq	%ebx, %rbx
	vmulsd	(%rdi,%rbx,8), %xmm3, %xmm9
	leal	8(%r13,%rsi), %ebx
	movslq	%ebx, %rbx
	vmulsd	(%rdi,%rbx,8), %xmm3, %xmm10
	vmovsd	-200(%rbp), %xmm5       ## xmm5 = mem[0],zero
	leal	1(%r13,%rsi), %ebx
	movslq	%ebx, %rbx
	vmulsd	(%rdi,%rbx,8), %xmm5, %xmm11
	leal	7(%r13,%rsi), %ebx
	movslq	%ebx, %rbx
	vmulsd	(%rdi,%rbx,8), %xmm5, %xmm13
	vmovsd	-192(%rbp), %xmm7       ## xmm7 = mem[0],zero
	leal	2(%r13,%rsi), %ebx
	movslq	%ebx, %rbx
	vmulsd	(%rdi,%rbx,8), %xmm7, %xmm14
	leal	6(%r13,%rsi), %ebx
	movslq	%ebx, %rbx
	vmulsd	(%rdi,%rbx,8), %xmm7, %xmm7
	vmovsd	-184(%rbp), %xmm1       ## xmm1 = mem[0],zero
	leal	3(%r13,%rsi), %ebx
	movslq	%ebx, %rbx
	vmulsd	(%rdi,%rbx,8), %xmm1, %xmm2
	leal	5(%r13,%rsi), %ebx
	movslq	%ebx, %rbx
	vmulsd	(%rdi,%rbx,8), %xmm1, %xmm1
	vmovq	%r12, %xmm3
	vmovq	%rdx, %xmm4
	movq	-304(%rbp), %rcx        ## 8-byte Reload
	movq	-296(%rbp), %rdx        ## 8-byte Reload
	movq	-288(%rbp), %r15        ## 8-byte Reload
	vmovq	%r10, %xmm0
	movq	-224(%rbp), %rbx        ## 8-byte Reload
	vmovq	%rax, %xmm5
	vmulsd	-216(%rbp), %xmm8, %xmm6
	vaddsd	(%r8,%r9,8), %xmm6, %xmm6
	vaddsd	%xmm9, %xmm6, %xmm6
	vaddsd	%xmm10, %xmm6, %xmm6
	vaddsd	%xmm11, %xmm6, %xmm6
	vaddsd	%xmm13, %xmm6, %xmm6
	vaddsd	%xmm14, %xmm6, %xmm6
	vaddsd	%xmm7, %xmm6, %xmm6
	vaddsd	%xmm2, %xmm6, %xmm2
	vaddsd	%xmm1, %xmm2, %xmm1
	vmulsd	-176(%rbp), %xmm3, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm1
	vmulsd	-168(%rbp), %xmm4, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm1
	vmovsd	-160(%rbp), %xmm2       ## xmm2 = mem[0],zero
	vmulsd	-64(%rbp), %xmm2, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm1
	vmovsd	-152(%rbp), %xmm2       ## xmm2 = mem[0],zero
	vmulsd	-56(%rbp), %xmm2, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm1
	vmulsd	-144(%rbp), %xmm12, %xmm2
	vaddsd	%xmm2, %xmm1, %xmm1
	vmulsd	-136(%rbp), %xmm0, %xmm0
	vaddsd	%xmm0, %xmm1, %xmm0
	vmulsd	-128(%rbp), %xmm5, %xmm1
	vaddsd	%xmm1, %xmm0, %xmm0
	vmovsd	-120(%rbp), %xmm1       ## xmm1 = mem[0],zero
	vmulsd	-88(%rbp), %xmm1, %xmm1
	vaddsd	%xmm1, %xmm0, %xmm0
	vmovsd	%xmm0, (%r8,%r9,8)
	vmovsd	-176(%rbp), %xmm0       ## xmm0 = mem[0],zero
	vmulsd	-48(%rbp), %xmm0, %xmm0
	vaddsd	(%rbx,%r14,8), %xmm0, %xmm0
	vmovsd	-168(%rbp), %xmm1       ## xmm1 = mem[0],zero
	vmulsd	-56(%rbp), %xmm1, %xmm1
	vaddsd	%xmm1, %xmm0, %xmm0
	vmovsd	-160(%rbp), %xmm1       ## xmm1 = mem[0],zero
	vmulsd	-64(%rbp), %xmm1, %xmm1
	vaddsd	%xmm1, %xmm0, %xmm0
	vmovsd	-152(%rbp), %xmm1       ## xmm1 = mem[0],zero
	vmulsd	-72(%rbp), %xmm1, %xmm1
	vaddsd	%xmm1, %xmm0, %xmm0
	vmovsd	%xmm0, (%rbx,%r14,8)
	movq	-280(%rbp), %r10        ## 8-byte Reload
	vmovsd	-144(%rbp), %xmm0       ## xmm0 = mem[0],zero
	vmulsd	-48(%rbp), %xmm0, %xmm0
	vmovsd	-136(%rbp), %xmm1       ## xmm1 = mem[0],zero
	vmulsd	-88(%rbp), %xmm1, %xmm1
	vaddsd	%xmm1, %xmm0, %xmm0
	vmovsd	-128(%rbp), %xmm1       ## xmm1 = mem[0],zero
	vmulsd	-96(%rbp), %xmm1, %xmm1
	vaddsd	%xmm1, %xmm0, %xmm0
	vmovsd	-120(%rbp), %xmm1       ## xmm1 = mem[0],zero
	vmulsd	-104(%rbp), %xmm1, %xmm1
	vaddsd	%xmm1, %xmm0, %xmm0
	movq	-232(%rbp), %rax        ## 8-byte Reload
	vmovsd	%xmm0, (%rax,%r9,8)
	incl	%esi
	incq	%r9
	cmpl	%r9d, %r11d
	jne	LBB0_62
## BB#63:                               ## %._crit_edge
                                        ##   in Loop: Header=BB0_61 Depth=2
	movq	-352(%rbp), %rax        ## 8-byte Reload
	movl	-308(%rbp), %r14d       ## 4-byte Reload
	addl	%eax, %r14d
	movl	-332(%rbp), %esi        ## 4-byte Reload
	addl	%eax, %esi
	movl	-328(%rbp), %ebx        ## 4-byte Reload
	addl	%eax, %ebx
	movl	-324(%rbp), %r12d       ## 4-byte Reload
	addl	%eax, %r12d
	movl	-320(%rbp), %r9d        ## 4-byte Reload
	addl	%eax, %r9d
	movl	-316(%rbp), %r8d        ## 4-byte Reload
	addl	%eax, %r8d
	movl	-312(%rbp), %eax        ## 4-byte Reload
	cmpl	-336(%rbp), %eax        ## 4-byte Folded Reload
	jne	LBB0_61
## BB#64:                               ## %._crit_edge.12
                                        ##   in Loop: Header=BB0_60 Depth=1
	movl	-388(%rbp), %eax        ## 4-byte Reload
	movq	-384(%rbp), %rdi        ## 8-byte Reload
	addl	%eax, %edi
	movl	-356(%rbp), %r8d        ## 4-byte Reload
	addl	%eax, %r8d
	movl	-360(%rbp), %r9d        ## 4-byte Reload
	addl	%eax, %r9d
	movl	-364(%rbp), %r12d       ## 4-byte Reload
	addl	%eax, %r12d
	movl	-368(%rbp), %ebx        ## 4-byte Reload
	addl	%eax, %ebx
	movl	-372(%rbp), %esi        ## 4-byte Reload
	addl	%eax, %esi
	movl	%esi, %eax
	movl	-376(%rbp), %esi        ## 4-byte Reload
	cmpl	-336(%rbp), %esi        ## 4-byte Folded Reload
	jne	LBB0_60
LBB0_65:                                ## %._crit_edge.15
	xorl	%eax, %eax
	addq	$360, %rsp              ## imm = 0x168
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	retq
	.cfi_endproc

	.section	__TEXT,__literal16,16byte_literals
	.align	4                       ## @.memset_pattern.1
L_.memset_pattern.1:
	.quad	4597454643604897137     ## double 2.300000e-01
	.quad	4597454643604897137     ## double 2.300000e-01


.subsections_via_symbols
