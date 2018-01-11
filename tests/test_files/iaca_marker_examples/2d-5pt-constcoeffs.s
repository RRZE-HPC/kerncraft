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
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	movq	%rsi, %rbx
	subq	$72, %rsp
	.cfi_def_cfa_offset 128
	movq	16(%rsi), %rdi
	xorl	%esi, %esi
	call	strtol
	movq	8(%rbx), %rdi
	xorl	%esi, %esi
	movl	$10, %edx
	movq	%rax, %r13
	movl	%eax, %ebp
	call	strtol
	leaq	48(%rsp), %rdi
	movl	$32, %esi
	movl	%eax, %r14d
	movq	%rax, %rbx
	imull	%r13d, %r14d
	movslq	%r14d, %r15
	salq	$3, %r15
	movq	%r15, %rdx
	call	posix_memalign
	testl	%eax, %eax
	je	.L2
	movq	$0, 48(%rsp)
.L2:
	testl	%r14d, %r14d
	movq	48(%rsp), %r12
	jle	.L13
	movq	%r12, %rax
	salq	$60, %rax
	shrq	$63, %rax
	cmpl	%eax, %r14d
	cmovbe	%r14d, %eax
	cmpl	$6, %r14d
	movl	%eax, %edx
	movl	%r14d, %eax
	ja	.L86
.L6:
	movsd	.LC0(%rip), %xmm5
	cmpl	$1, %eax
	movsd	%xmm5, (%r12)
	jbe	.L43
	cmpl	$2, %eax
	movsd	%xmm5, 8(%r12)
	jbe	.L44
	cmpl	$3, %eax
	movsd	%xmm5, 16(%r12)
	jbe	.L45
	cmpl	$4, %eax
	movsd	%xmm5, 24(%r12)
	jbe	.L46
	cmpl	$5, %eax
	movsd	%xmm5, 32(%r12)
	jbe	.L47
	movsd	%xmm5, 40(%r12)
	movl	$6, %esi
.L8:
	cmpl	%r14d, %eax
	je	.L13
.L7:
	movl	%r14d, %edi
	movl	%eax, %edx
	subl	%eax, %edi
	movl	%edi, %ecx
	shrl	%ecx
	movl	%ecx, %r8d
	addl	%r8d, %r8d
	je	.L10
	movapd	.LC1(%rip), %xmm0
	leaq	(%r12,%rdx,8), %rdx
	xorl	%eax, %eax
.L14:
	addl	$1, %eax
	movapd	%xmm0, (%rdx)
	addq	$16, %rdx
	cmpl	%eax, %ecx
	ja	.L14
	addl	%r8d, %esi
	cmpl	%edi, %r8d
	je	.L13
.L10:
	movsd	.LC0(%rip), %xmm2
	movslq	%esi, %rsi
	movsd	%xmm2, (%r12,%rsi,8)
.L13:
	movl	var_false(%rip), %ecx
	testl	%ecx, %ecx
	jne	.L87
.L5:
	leaq	48(%rsp), %rdi
	movq	%r15, %rdx
	movl	$32, %esi
	call	posix_memalign
	testl	%eax, %eax
	je	.L15
	movq	$0, 48(%rsp)
.L15:
	movq	48(%rsp), %rax
	testl	%r14d, %r14d
	movq	%rax, (%rsp)
	jle	.L26
	salq	$60, %rax
	shrq	$63, %rax
	cmpl	%eax, %r14d
	cmovbe	%r14d, %eax
	cmpl	$6, %r14d
	movl	%eax, %edx
	movl	%r14d, %eax
	ja	.L88
.L19:
	movq	(%rsp), %rdi
	cmpl	$1, %eax
	movsd	.LC2(%rip), %xmm2
	movsd	%xmm2, (%rdi)
	jbe	.L50
	cmpl	$2, %eax
	movsd	%xmm2, 8(%rdi)
	jbe	.L51
	cmpl	$3, %eax
	movsd	%xmm2, 16(%rdi)
	jbe	.L52
	cmpl	$4, %eax
	movsd	%xmm2, 24(%rdi)
	jbe	.L53
	cmpl	$5, %eax
	movsd	%xmm2, 32(%rdi)
	jbe	.L54
	movsd	%xmm2, 40(%rdi)
	movl	$6, %esi
.L21:
	cmpl	%eax, %r14d
	je	.L26
.L20:
	subl	%eax, %r14d
	movl	%eax, %edx
	movl	%r14d, %ecx
	shrl	%ecx
	movl	%ecx, %edi
	addl	%edi, %edi
	je	.L23
	movq	(%rsp), %rax
	movapd	.LC3(%rip), %xmm0
	leaq	(%rax,%rdx,8), %rdx
	xorl	%eax, %eax
.L27:
	addl	$1, %eax
	movapd	%xmm0, (%rdx)
	addq	$16, %rdx
	cmpl	%eax, %ecx
	ja	.L27
	addl	%edi, %esi
	cmpl	%r14d, %edi
	je	.L26
.L23:
	movq	(%rsp), %rax
	movslq	%esi, %rsi
	movsd	.LC2(%rip), %xmm3
	movsd	%xmm3, (%rax,%rsi,8)
.L26:
	movl	var_false(%rip), %edx
	testl	%edx, %edx
	jne	.L89
	movsd	.LC4(%rip), %xmm2
	movsd	%xmm2, 16(%rsp)
.L28:
	movsd	.LC5(%rip), %xmm3
	movsd	%xmm3, 32(%rsp)
.L29:
	movsd	.LC6(%rip), %xmm4
	movl	$0, 12(%rsp)
	movsd	%xmm4, 48(%rsp)
.L30:
	leal	-1(%rbx), %edi
	cmpl	$1, %edi
	jle	.L31
	movslq	%r13d, %rdx
	movq	%r12, %r10
	movl	$2, %r11d
	leal	-3(%r13), %eax
	leaq	0(,%rdx,8), %rbx
	addq	%rax, %rdx
	salq	$3, %rax
	leaq	16(%r12,%rdx,8), %r9
	leal	(%r13,%r13), %edx
	movq	(%rsp), %r13
	leaq	8(%rbx), %r15
	movslq	%edx, %rdx
	leaq	8(,%rdx,8), %r14
	subq	%rax, %r13
	subq	%r12, %r13
	subq	$8, %r13
	.p2align 4,,10
	.p2align 3
.L32:
	cmpl	$2, %ebp
	movl	%r11d, %r8d
	jle	.L35
	leaq	0(%r13,%r9), %rsi
	movq	%r10, %rdx
	leaq	(%r14,%r10), %rcx
	leaq	(%r15,%r10), %rax
	.p2align 4,,10
	.p2align 3
.L36:
	movsd	-8(%rax), %xmm0
	addq	$8, %rax
	addq	$8, %rdx
	movsd	-8(%rax), %xmm1
	addq	$8, %rcx
	addq	$8, %rsi
	addsd	(%rdx), %xmm0
	mulsd	16(%rsp), %xmm1
	addsd	-8(%rcx), %xmm0
	addsd	(%rax), %xmm0
	mulsd	32(%rsp), %xmm0
	addsd	%xmm0, %xmm1
	movsd	-8(%rdx), %xmm0
	addsd	-16(%rcx), %xmm0
	addsd	8(%rdx), %xmm0
	addsd	(%rcx), %xmm0
	mulsd	48(%rsp), %xmm0
	addsd	%xmm0, %xmm1
	movsd	%xmm1, -8(%rsi)
	cmpq	%rax, %r9
	jne	.L36
.L35:
	addl	$1, %r11d
	addq	%rbx, %r10
	addq	%rbx, %r9
	cmpl	%r8d, %edi
	jg	.L32
.L31:
	movl	12(%rsp), %eax
	testl	%eax, %eax
	jne	.L90
.L63:
	addq	$72, %rsp
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
.L88:
	.cfi_restore_state
	xorl	%eax, %eax
	xorl	%esi, %esi
	testl	%edx, %edx
	je	.L20
	movl	%edx, %eax
	jmp	.L19
.L86:
	xorl	%eax, %eax
	xorl	%esi, %esi
	testl	%edx, %edx
	je	.L7
	movl	%edx, %eax
	jmp	.L6
.L87:
	movq	%r12, %rdi
	call	dummy
	.p2align 4,,3
	jmp	.L5
.L90:
	movq	%r12, %rdi
	call	dummy
	cmpl	$0, var_false(%rip)
	je	.L63
	movq	(%rsp), %rdi
	call	dummy
	cmpl	$0, var_false(%rip)
	je	.L63
	leaq	16(%rsp), %rdi
	call	dummy
	cmpl	$0, var_false(%rip)
	je	.L63
	leaq	32(%rsp), %rdi
	call	dummy
	cmpl	$0, var_false(%rip)
	je	.L63
	leaq	48(%rsp), %rdi
	call	dummy
	jmp	.L63
.L89:
	movq	(%rsp), %rdi
	call	dummy
	cmpl	$0, var_false(%rip)
	movsd	.LC4(%rip), %xmm4
	movsd	%xmm4, 16(%rsp)
	je	.L28
	leaq	16(%rsp), %rdi
	call	dummy
	cmpl	$0, var_false(%rip)
	movsd	.LC5(%rip), %xmm4
	movsd	%xmm4, 32(%rsp)
	je	.L29
	leaq	32(%rsp), %rdi
	call	dummy
	movl	var_false(%rip), %eax
	movsd	.LC6(%rip), %xmm2
	movsd	%xmm2, 48(%rsp)
	testl	%eax, %eax
	movl	%eax, 12(%rsp)
	je	.L30
	leaq	48(%rsp), %rdi
	call	dummy
	movl	var_false(%rip), %eax
	movl	%eax, 12(%rsp)
	jmp	.L30
.L43:
	movl	$1, %esi
	jmp	.L8
.L44:
	movl	$2, %esi
	jmp	.L8
.L45:
	movl	$3, %esi
	jmp	.L8
.L46:
	movl	$4, %esi
	jmp	.L8
.L47:
	movl	$5, %esi
	jmp	.L8
.L51:
	movl	$2, %esi
	jmp	.L21
.L52:
	movl	$3, %esi
	jmp	.L21
.L53:
	movl	$4, %esi
	jmp	.L21
.L54:
	movl	$5, %esi
	jmp	.L21
.L50:
	movl	$1, %esi
	jmp	.L21
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
	.align 8
.LC5:
	.long	765859228
	.long	1071838070
	.align 8
.LC6:
	.long	2226626236
	.long	1072102945
	.ident	"GCC: (GNU) 4.8.5 20150623 (Red Hat 4.8.5-4)"
	.section	.note.GNU-stack,"",@progbits
