	.file	"stencil.c_compilable.c"
	.section	.text.startup,"ax",@progbits
	.p2align 4,,15
	.globl	main
	.type	main, @function
main:
.LFB5:
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	movl	$10, %edx
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	movq	%rsi, %rbp
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$200, %rsp
	.cfi_def_cfa_offset 256
	movq	16(%rsi), %rdi
	xorl	%esi, %esi
	call	strtol
	movq	8(%rbp), %rdi
	xorl	%esi, %esi
	movl	$10, %edx
	movq	%rax, %rbx
	movl	%eax, 104(%rsp)
	call	strtol
	movl	$32, %esi
	leaq	176(%rsp), %rdi
	movl	%eax, %r12d
	movq	%rax, %rbp
	imull	%ebx, %r12d
	movslq	%r12d, %r13
	salq	$3, %r13
	movq	%r13, %rdx
	call	posix_memalign
	testl	%eax, %eax
	je	.L2
	movq	$0, 176(%rsp)
.L2:
	movq	176(%rsp), %rax
	testl	%r12d, %r12d
	movq	%rax, 120(%rsp)
	jle	.L13
	salq	$60, %rax
	shrq	$63, %rax
	cmpl	%eax, %r12d
	cmovbe	%r12d, %eax
	cmpl	$6, %r12d
	movl	%eax, %edx
	movl	%r12d, %eax
	ja	.L109
.L6:
	movq	120(%rsp), %rcx
	cmpl	$1, %eax
	movsd	.LC0(%rip), %xmm5
	movsd	%xmm5, (%rcx)
	jbe	.L58
	cmpl	$2, %eax
	movsd	%xmm5, 8(%rcx)
	jbe	.L59
	cmpl	$3, %eax
	movsd	%xmm5, 16(%rcx)
	jbe	.L60
	cmpl	$4, %eax
	movsd	%xmm5, 24(%rcx)
	jbe	.L61
	cmpl	$5, %eax
	movsd	%xmm5, 32(%rcx)
	jbe	.L62
	movsd	%xmm5, 40(%rcx)
	movl	$6, %esi
.L8:
	cmpl	%eax, %r12d
	je	.L13
.L7:
	movl	%r12d, %edi
	movl	%eax, %edx
	subl	%eax, %edi
	movl	%edi, %ecx
	shrl	%ecx
	movl	%ecx, %r8d
	addl	%r8d, %r8d
	je	.L10
	movq	120(%rsp), %rax
	movapd	.LC1(%rip), %xmm0
	leaq	(%rax,%rdx,8), %rdx
	xorl	%eax, %eax
.L14:
	addl	$1, %eax
	movapd	%xmm0, (%rdx)
	addq	$16, %rdx
	cmpl	%ecx, %eax
	jb	.L14
	addl	%r8d, %esi
	cmpl	%r8d, %edi
	je	.L13
.L10:
	movq	120(%rsp), %rax
	movslq	%esi, %rsi
	movsd	.LC0(%rip), %xmm5
	movsd	%xmm5, (%rax,%rsi,8)
.L13:
	movl	var_false(%rip), %r14d
	testl	%r14d, %r14d
	jne	.L110
.L5:
	leaq	176(%rsp), %rdi
	movq	%r13, %rdx
	movl	$32, %esi
	call	posix_memalign
	testl	%eax, %eax
	je	.L15
	movq	$0, 176(%rsp)
.L15:
	movq	176(%rsp), %rax
	testl	%r12d, %r12d
	movq	%rax, 128(%rsp)
	jle	.L26
	salq	$60, %rax
	shrq	$63, %rax
	cmpl	%eax, %r12d
	cmovbe	%r12d, %eax
	cmpl	$6, %r12d
	movl	%eax, %edx
	movl	%r12d, %eax
	ja	.L111
.L19:
	movq	128(%rsp), %rsi
	cmpl	$1, %eax
	movsd	.LC2(%rip), %xmm5
	movsd	%xmm5, (%rsi)
	jbe	.L65
	cmpl	$2, %eax
	movsd	%xmm5, 8(%rsi)
	jbe	.L66
	cmpl	$3, %eax
	movsd	%xmm5, 16(%rsi)
	jbe	.L67
	cmpl	$4, %eax
	movsd	%xmm5, 24(%rsi)
	jbe	.L68
	cmpl	$5, %eax
	movsd	%xmm5, 32(%rsi)
	jbe	.L69
	movsd	%xmm5, 40(%rsi)
	movl	$6, %esi
.L21:
	cmpl	%eax, %r12d
	je	.L26
.L20:
	movl	%r12d, %edi
	movl	%eax, %edx
	subl	%eax, %edi
	movl	%edi, %ecx
	shrl	%ecx
	movl	%ecx, %r8d
	addl	%r8d, %r8d
	je	.L23
	movq	128(%rsp), %rax
	movapd	.LC3(%rip), %xmm0
	leaq	(%rax,%rdx,8), %rdx
	xorl	%eax, %eax
.L27:
	addl	$1, %eax
	movapd	%xmm0, (%rdx)
	addq	$16, %rdx
	cmpl	%ecx, %eax
	jb	.L27
	addl	%r8d, %esi
	cmpl	%r8d, %edi
	je	.L26
.L23:
	movq	128(%rsp), %rax
	movslq	%esi, %rsi
	movsd	.LC2(%rip), %xmm6
	movsd	%xmm6, (%rax,%rsi,8)
.L26:
	movl	var_false(%rip), %r13d
	testl	%r13d, %r13d
	jne	.L112
.L18:
	leaq	176(%rsp), %rdi
	addl	%r12d, %r12d
	movl	$32, %esi
	movslq	%r12d, %rdx
	salq	$3, %rdx
	call	posix_memalign
	testl	%eax, %eax
	je	.L28
	movq	$0, 176(%rsp)
.L28:
	movq	176(%rsp), %rax
	testl	%r12d, %r12d
	movq	%rax, 48(%rsp)
	jle	.L39
	salq	$60, %rax
	shrq	$63, %rax
	cmpl	%eax, %r12d
	cmovbe	%r12d, %eax
	cmpl	$6, %r12d
	movl	%eax, %edx
	movl	%r12d, %eax
	ja	.L113
.L32:
	movq	48(%rsp), %rdi
	cmpl	$1, %eax
	movsd	.LC4(%rip), %xmm5
	movsd	%xmm5, (%rdi)
	jbe	.L72
	cmpl	$2, %eax
	movsd	%xmm5, 8(%rdi)
	jbe	.L73
	cmpl	$3, %eax
	movsd	%xmm5, 16(%rdi)
	jbe	.L74
	cmpl	$4, %eax
	movsd	%xmm5, 24(%rdi)
	jbe	.L75
	cmpl	$5, %eax
	movsd	%xmm5, 32(%rdi)
	jbe	.L76
	movsd	%xmm5, 40(%rdi)
	movl	$6, %esi
.L34:
	cmpl	%eax, %r12d
	je	.L39
.L33:
	subl	%eax, %r12d
	movl	%eax, %edx
	movl	%r12d, %ecx
	shrl	%ecx
	movl	%ecx, %edi
	addl	%edi, %edi
	je	.L36
	movq	48(%rsp), %rax
	movapd	.LC5(%rip), %xmm0
	leaq	(%rax,%rdx,8), %rdx
	xorl	%eax, %eax
.L40:
	addl	$1, %eax
	movapd	%xmm0, (%rdx)
	addq	$16, %rdx
	cmpl	%eax, %ecx
	ja	.L40
	addl	%edi, %esi
	cmpl	%edi, %r12d
	je	.L39
.L36:
	movq	48(%rsp), %rax
	movslq	%esi, %rsi
	movsd	.LC4(%rip), %xmm7
	movsd	%xmm7, (%rax,%rsi,8)
.L39:
	movl	var_false(%rip), %r12d
	testl	%r12d, %r12d
	jne	.L30
	movl	$0, 172(%rsp)
.L31:
	leal	-1(%rbp), %eax
	cmpl	$1, %eax
	movl	%eax, 108(%rsp)
	jle	.L41
	movq	48(%rsp), %r11
	leal	(%rbx,%rbx), %eax
	movl	%ebx, 136(%rsp)
	movl	%eax, %r12d
	movl	%eax, 40(%rsp)
	cltq
	movq	128(%rsp), %rbp
	movq	%rax, %r9
	movl	%r12d, 44(%rsp)
	leal	-3(%rbx), %r8d
	movl	%ebx, 140(%rsp)
	movq	120(%rsp), %r13
	leaq	(%r9,%r8,2), %r8
	movq	%rax, 160(%rsp)
	movq	%r11, %rcx
	salq	$3, %rax
	movq	$0, 80(%rsp)
	movq	%rax, 64(%rsp)
	addq	%rax, %rcx
	movslq	%ebx, %rax
	leaq	32(%r11,%r8,8), %r8
	subl	$2, %ebx
	movq	%rax, %r10
	leaq	0(,%rax,8), %rdi
	movq	%rax, 112(%rsp)
	movl	%ebx, %r15d
	addq	$1, %rax
	movq	%r8, 88(%rsp)
	movl	%ebx, %r8d
	leaq	2(%r9,%r8,2), %r9
	movl	%ebx, 56(%rsp)
	movq	%rbp, %r14
	leaq	0(%rbp,%rax,8), %rsi
	addq	%r8, %rax
	movq	%r13, %r8
	leaq	(%r11,%r9,8), %rbx
	addq	%rdi, %r14
	addq	64(%rsp), %r8
	movq	%rbx, 16(%rsp)
	movq	%r10, %rbx
	leaq	0(%rbp,%rax,8), %r9
	movl	%r15d, %eax
	movl	$1, %r15d
	leaq	24(%rbp,%r10,8), %rbp
	shrl	%eax
	movq	%r9, 24(%rsp)
	movl	%eax, 168(%rsp)
	addl	%eax, %eax
	movq	%r13, %r9
	movl	%eax, 96(%rsp)
	leaq	0(%r13,%rdi), %rdx
	movl	%eax, %r10d
	addl	$1, %eax
	testl	%r10d, %r10d
	movq	%rbx, 72(%rsp)
	cmove	%r15d, %eax
	movl	$2, 8(%rsp)
	leal	(%rax,%rax), %r10d
	movl	%eax, 100(%rsp)
	movl	%r10d, 156(%rsp)
	addl	$1, %r10d
	movl	%r10d, 152(%rsp)
	leal	-1(%rax), %r10d
	addl	$1, %eax
	movl	%r10d, 148(%rsp)
	xorl	%r10d, %r10d
	movl	%eax, 144(%rsp)
	.p2align 4,,10
	.p2align 3
.L42:
	cmpl	$2, 104(%rsp)
	jle	.L114
	movl	140(%rsp), %eax
	leaq	8(%r8), %r11
	leaq	8(%r9), %r15
	leaq	16(%rdx), %rbx
	leaq	8(%rdx), %r12
	leal	(%rax,%r10), %r13d
	movl	8(%rsp), %eax
	movl	%eax, 12(%rsp)
	movl	40(%rsp), %eax
	addl	%r10d, %eax
	movl	%eax, 60(%rsp)
	leaq	16(%rcx), %rax
	movq	%rax, 32(%rsp)
	leaq	24(%r8), %rax
	cmpq	%rax, %rsi
	setnb	%al
	cmpq	%r11, %rbp
	setbe	%r11b
	orl	%r11d, %eax
	leaq	24(%r9), %r11
	cmpq	%r11, %rsi
	setnb	%r11b
	cmpq	%r15, %rbp
	setbe	%r15b
	orl	%r15d, %r11d
	andl	%r11d, %eax
	cmpl	$8, 56(%rsp)
	seta	%r11b
	andl	%r11d, %eax
	leaq	32(%rdx), %r11
	cmpq	%r11, %rsi
	setnb	%r11b
	cmpq	%rbx, %rbp
	setbe	%r15b
	orl	%r15d, %r11d
	andl	%r11d, %eax
	cmpq	%rbx, %rsi
	setnb	%r11b
	cmpq	%rdx, %rbp
	setbe	%bl
	orl	%ebx, %r11d
	andl	%r11d, %eax
	leaq	24(%rdx), %r11
	cmpq	%r11, %rsi
	setnb	%r11b
	cmpq	%r12, %rbp
	setbe	%bl
	orl	%ebx, %r11d
	testb	%r11b, %al
	je	.L53
	movq	32(%rsp), %rax
	cmpq	%rax, 24(%rsp)
	setbe	%r11b
	cmpq	16(%rsp), %rsi
	setnb	%al
	orb	%al, %r11b
	je	.L53
	movl	96(%rsp), %r11d
	testl	%r11d, %r11d
	je	.L46
	movl	168(%rsp), %ebx
	xorl	%eax, %eax
	xorl	%r11d, %r11d
.L43:
	movupd	16(%rcx,%rax,2), %xmm2
	addl	$1, %r11d
	movupd	32(%rcx,%rax,2), %xmm1
	movupd	8(%rdx,%rax), %xmm3
	movapd	%xmm2, %xmm0
	unpcklpd	%xmm1, %xmm0
	unpckhpd	%xmm1, %xmm2
	movupd	(%rdx,%rax), %xmm1
	mulpd	%xmm3, %xmm0
	movupd	16(%rdx,%rax), %xmm3
	movupd	8(%r8,%rax), %xmm4
	addpd	%xmm3, %xmm1
	movupd	8(%r9,%rax), %xmm3
	addpd	%xmm4, %xmm3
	addpd	%xmm3, %xmm1
	mulpd	%xmm1, %xmm2
	addpd	%xmm2, %xmm0
	movupd	%xmm0, 8(%r14,%rax)
	addq	$16, %rax
	cmpl	%r11d, %ebx
	ja	.L43
	movl	56(%rsp), %eax
	cmpl	%eax, 96(%rsp)
	je	.L50
.L46:
	movl	44(%rsp), %r12d
	movl	156(%rsp), %r11d
	movq	48(%rsp), %r15
	movl	100(%rsp), %eax
	addl	%r12d, %r11d
	movslq	%r11d, %r11
	movsd	(%r15,%r11,8), %xmm1
	addl	%r13d, %eax
	movl	152(%rsp), %r11d
	cltq
	movq	120(%rsp), %r15
	addl	%r12d, %r11d
	movl	148(%rsp), %r12d
	mulsd	(%r15,%rax,8), %xmm1
	movslq	%r11d, %r11
	leal	(%r12,%r13), %ebx
	movl	144(%rsp), %r12d
	movslq	%ebx, %rbx
	movsd	(%r15,%rbx,8), %xmm0
	movl	100(%rsp), %ebx
	addl	%r13d, %r12d
	movl	60(%rsp), %r13d
	movslq	%r12d, %r12
	addsd	(%r15,%r12,8), %xmm0
	leal	(%r10,%rbx), %r12d
	movslq	%r12d, %r12
	addl	%ebx, %r13d
	movsd	(%r15,%r12,8), %xmm2
	movslq	%r13d, %rbx
	addsd	(%r15,%rbx,8), %xmm2
	movq	48(%rsp), %r15
	movq	128(%rsp), %rbx
	addsd	%xmm2, %xmm0
	mulsd	(%r15,%r11,8), %xmm0
	addsd	%xmm0, %xmm1
	movsd	%xmm1, (%rbx,%rax,8)
.L50:
	movq	64(%rsp), %rax
	addq	%rdi, %rdx
	addq	%rdi, %rsi
	movq	112(%rsp), %rbx
	addq	%rdi, %r14
	addq	%rdi, %rbp
	addl	$1, 8(%rsp)
	addq	%rdi, %r9
	addq	%rdi, %r8
	addq	%rdi, 24(%rsp)
	addq	%rax, %rcx
	addl	136(%rsp), %r10d
	addq	%rax, 88(%rsp)
	addq	%rax, 16(%rsp)
	movl	40(%rsp), %eax
	addq	%rbx, 80(%rsp)
	addq	%rbx, 72(%rsp)
	addl	%eax, 44(%rsp)
	movl	12(%rsp), %eax
	cmpl	%eax, 108(%rsp)
	jg	.L42
.L41:
	movl	172(%rsp), %eax
	testl	%eax, %eax
	jne	.L115
.L55:
	addq	$200, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	xorl	%eax, %eax
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L53:
	.cfi_restore_state
	movq	80(%rsp), %rbx
	movq	%rsi, %rax
	movq	72(%rsp), %r15
	movq	32(%rsp), %r13
	movq	%rbx, %r11
	addq	160(%rsp), %r11
	subq	%r15, %rbx
	subq	%r15, %r11
	movq	88(%rsp), %r15
	.p2align 4,,10
	.p2align 3
.L45:
	movsd	-8(%r12), %xmm0
	addq	$16, %r13
	addq	$8, %rax
	movsd	(%r12,%rbx,8), %xmm2
	addsd	8(%r12), %xmm0
	addsd	(%r12,%r11,8), %xmm2
	addq	$8, %r12
	movsd	-16(%r13), %xmm1
	addsd	%xmm2, %xmm0
	mulsd	-8(%r12), %xmm1
	mulsd	-8(%r13), %xmm0
	addsd	%xmm0, %xmm1
	movsd	%xmm1, -8(%rax)
	cmpq	%r15, %r13
	jne	.L45
	jmp	.L50
	.p2align 4,,10
	.p2align 3
.L114:
	movl	8(%rsp), %eax
	movl	%eax, 12(%rsp)
	jmp	.L50
.L113:
	xorl	%eax, %eax
	xorl	%esi, %esi
	testl	%edx, %edx
	je	.L33
	movl	%edx, %eax
	jmp	.L32
.L111:
	xorl	%eax, %eax
	xorl	%esi, %esi
	testl	%edx, %edx
	je	.L20
	movl	%edx, %eax
	jmp	.L19
.L109:
	xorl	%eax, %eax
	xorl	%esi, %esi
	testl	%edx, %edx
	je	.L7
	movl	%edx, %eax
	jmp	.L6
.L115:
	movq	120(%rsp), %rdi
	call	dummy
	cmpl	$0, var_false(%rip)
	je	.L55
	movq	128(%rsp), %rdi
	call	dummy
	cmpl	$0, var_false(%rip)
	je	.L55
	movq	48(%rsp), %rdi
	call	dummy
	jmp	.L55
.L30:
	movq	48(%rsp), %rdi
	call	dummy
	movl	var_false(%rip), %eax
	movl	%eax, 172(%rsp)
	jmp	.L31
.L112:
	movq	128(%rsp), %rdi
	call	dummy
	jmp	.L18
.L110:
	movq	120(%rsp), %rdi
	call	dummy
	.p2align 4,,3
	jmp	.L5
.L58:
	movl	$1, %esi
	jmp	.L8
.L59:
	movl	$2, %esi
	jmp	.L8
.L60:
	movl	$3, %esi
	jmp	.L8
.L61:
	movl	$4, %esi
	jmp	.L8
.L62:
	movl	$5, %esi
	jmp	.L8
.L73:
	movl	$2, %esi
	jmp	.L34
.L74:
	movl	$3, %esi
	jmp	.L34
.L75:
	movl	$4, %esi
	jmp	.L34
.L76:
	movl	$5, %esi
	jmp	.L34
.L66:
	movl	$2, %esi
	jmp	.L21
.L67:
	movl	$3, %esi
	jmp	.L21
.L68:
	movl	$4, %esi
	jmp	.L21
.L69:
	movl	$5, %esi
	jmp	.L21
.L65:
	movl	$1, %esi
	jmp	.L21
.L72:
	movl	$1, %esi
	jmp	.L34
	.cfi_endproc
.LFE5:
	.size	main, .-main
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC0:
	.long	3961705502
	.long	1071636094
	.section	.rodata.cst16,"aM",@progbits,16
	.align 16
.LC1:
	.long	3961705502
	.long	1071636094
	.long	3961705502
	.long	1071636094
	.section	.rodata.cst8
	.align 8
.LC2:
	.long	424680910
	.long	1071288493
	.section	.rodata.cst16
	.align 16
.LC3:
	.long	424680910
	.long	1071288493
	.long	424680910
	.long	1071288493
	.section	.rodata.cst8
	.align 8
.LC4:
	.long	3440069995
	.long	1072191488
	.section	.rodata.cst16
	.align 16
.LC5:
	.long	3440069995
	.long	1072191488
	.long	3440069995
	.long	1072191488
	.ident	"GCC: (GNU) 4.8.5 20150623 (Red Hat 4.8.5-4)"
	.section	.note.GNU-stack,"",@progbits
