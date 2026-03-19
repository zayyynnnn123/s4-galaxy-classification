	.file	"s4d.c"
	.text
	.p2align 4
	.globl	s4d_forward
	.type	s4d_forward, @function
s4d_forward:
.LFB40:
	.cfi_startproc
	endbr64
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
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
	leaq	-16384(%rsp), %r11
	.cfi_def_cfa 11, 16440
.LPSRL0:
	subq	$4096, %rsp
	orq	$0, (%rsp)
	cmpq	%r11, %rsp
	jne	.LPSRL0
	.cfi_def_cfa_register 7
	subq	$136, %rsp
	.cfi_def_cfa_offset 16576
	xorl	%r8d, %r8d
	movq	%rsi, 96(%rsp)
	leaq	112(%rsp), %rbp
	movq	%rdx, %rbx
	leaq	108(%rsp), %r14
	leaq	-256(%rdi), %r9
	movq	%fs:40, %rax
	movq	%rax, 16504(%rsp)
	xorl	%eax, %eax
	movslq	%ecx, %rax
	movl	%ecx, 60(%rsp)
	salq	$2, %rax
	movq	%rbp, 64(%rsp)
	movq	%rax, 88(%rsp)
	leaq	0(%rbp,%rax), %r15
	movq	%r15, %r13
	movq	%r9, %r15
.L6:
	movss	(%rbx,%r8,4), %xmm0
	movq	%r8, 8(%rsp)
	call	expf@PLT
	movq	88(%rsp), %rdx
	movq	64(%rsp), %rdi
	xorl	%esi, %esi
	movl	$16384, %ecx
	movss	%xmm0, 20(%rsp)
	call	__memset_chk@PLT
	movl	60(%rsp), %eax
	movq	8(%rsp), %r8
	testl	%eax, %eax
	jle	.L2
	movq	%r8, %rbp
	leaq	104(%rsp), %r12
	movq	%r8, 80(%rsp)
	salq	$7, %rbp
	movq	%r12, 48(%rsp)
	movq	%r13, %r12
	movq	64(%rsp), %r13
	leaq	128(%rbp), %rax
	movq	%r15, 72(%rsp)
	movq	%rbp, %r15
	movq	%rax, %rbp
	.p2align 4,,10
	.p2align 3
.L4:
	movss	256(%rbx,%r15), %xmm1
	movaps	%xmm1, %xmm0
	movss	%xmm1, 56(%rsp)
	call	expf@PLT
	movq	48(%rsp), %rsi
	movq	%r14, %rdi
	movss	8448(%rbx,%r15), %xmm2
	movaps	%xmm0, %xmm7
	movss	%xmm0, 44(%rsp)
	movss	20(%rsp), %xmm0
	xorps	.LC2(%rip), %xmm7
	movss	%xmm2, 40(%rsp)
	mulss	%xmm2, %xmm0
	movss	%xmm7, 8(%rsp)
	call	sincosf@PLT
	movss	20(%rsp), %xmm6
	mulss	8(%rsp), %xmm6
	movaps	%xmm6, %xmm0
	call	expf@PLT
	movss	104(%rsp), %xmm4
	movss	56(%rsp), %xmm1
	movss	16640(%rbx,%r15,2), %xmm6
	movaps	%xmm0, %xmm3
	movss	16644(%rbx,%r15,2), %xmm5
	mulss	%xmm0, %xmm4
	addss	%xmm1, %xmm1
	mulss	108(%rsp), %xmm3
	movss	%xmm6, 28(%rsp)
	movss	%xmm5, 24(%rsp)
	movaps	%xmm1, %xmm0
	movss	%xmm4, 36(%rsp)
	movss	%xmm3, 32(%rsp)
	call	expf@PLT
	movss	40(%rsp), %xmm2
	movq	%r13, %rax
	movss	.LC1(%rip), %xmm7
	movss	36(%rsp), %xmm4
	movss	44(%rsp), %xmm9
	movaps	%xmm2, %xmm8
	movaps	%xmm7, %xmm1
	movss	32(%rsp), %xmm3
	movaps	%xmm2, %xmm6
	mulss	%xmm2, %xmm8
	subss	%xmm4, %xmm1
	movss	24(%rsp), %xmm5
	mulss	%xmm3, %xmm6
	mulss	%xmm9, %xmm1
	movaps	%xmm4, %xmm9
	subss	%xmm7, %xmm9
	addss	%xmm0, %xmm8
	movss	8(%rsp), %xmm0
	mulss	%xmm9, %xmm2
	mulss	%xmm3, %xmm0
	addss	%xmm6, %xmm1
	movss	28(%rsp), %xmm6
	divss	%xmm8, %xmm1
	subss	%xmm2, %xmm0
	movaps	%xmm5, %xmm2
	divss	%xmm8, %xmm0
	movaps	%xmm6, %xmm8
	mulss	%xmm1, %xmm5
	mulss	%xmm1, %xmm8
	pxor	%xmm1, %xmm1
	mulss	%xmm0, %xmm6
	mulss	%xmm0, %xmm2
	addss	%xmm5, %xmm6
	subss	%xmm2, %xmm8
	movaps	%xmm7, %xmm2
	.p2align 4,,10
	.p2align 3
.L3:
	movaps	%xmm8, %xmm0
	movaps	%xmm6, %xmm5
	addq	$4, %rax
	mulss	%xmm1, %xmm5
	mulss	%xmm2, %xmm0
	subss	%xmm5, %xmm0
	movaps	%xmm3, %xmm5
	mulss	%xmm1, %xmm5
	mulss	%xmm4, %xmm1
	addss	%xmm0, %xmm0
	addss	-4(%rax), %xmm0
	movss	%xmm0, -4(%rax)
	movaps	%xmm2, %xmm0
	movaps	%xmm4, %xmm2
	mulss	%xmm0, %xmm2
	mulss	%xmm3, %xmm0
	subss	%xmm5, %xmm2
	addss	%xmm0, %xmm1
	cmpq	%rax, %r12
	jne	.L3
	addq	$4, %r15
	cmpq	%r15, %rbp
	jne	.L4
	movq	80(%rsp), %r8
	movq	72(%rsp), %r15
	movq	%r12, %r13
	xorl	%edi, %edi
	movq	96(%rsp), %rax
	movl	60(%rsp), %r10d
	xorl	%esi, %esi
	movss	33024(%rbx,%r8,4), %xmm2
	leaq	256(%r15), %rcx
	leaq	(%rax,%r8,4), %r12
	.p2align 4,,10
	.p2align 3
.L5:
	movss	(%rcx), %xmm1
	movq	%rcx, %rax
	movq	%r13, %rdx
	mulss	%xmm2, %xmm1
	.p2align 4,,10
	.p2align 3
.L8:
	movss	-4(%rdx), %xmm0
	mulss	(%rax), %xmm0
	subq	$256, %rax
	subq	$4, %rdx
	addss	%xmm0, %xmm1
	cmpq	%rax, %r15
	jne	.L8
	addl	$1, %esi
	movss	%xmm1, (%r12,%rdi)
	addq	$256, %rcx
	addq	$256, %rdi
	cmpl	%esi, %r10d
	jne	.L5
.L2:
	addq	$1, %r8
	addq	$4, %r15
	cmpq	$64, %r8
	jne	.L6
	movq	16504(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L18
	addq	$16520, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
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
.L18:
	.cfi_restore_state
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE40:
	.size	s4d_forward, .-s4d_forward
	.section	.rodata.cst4,"aM",@progbits,4
	.align 4
.LC1:
	.long	1065353216
	.section	.rodata.cst16,"aM",@progbits,16
	.align 16
.LC2:
	.long	-2147483648
	.long	0
	.long	0
	.long	0
	.ident	"GCC: (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
