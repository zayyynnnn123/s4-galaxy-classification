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
	movl	%ecx, %r15d
	leaq	108(%rsp), %r14
	movq	%rdx, %r12
	leaq	-256(%rdi), %r9
	movl	%r15d, %ebp
	movq	%r9, %r15
	movq	%fs:40, %rax
	movq	%rax, 16504(%rsp)
	xorl	%eax, %eax
	movslq	%ecx, %rax
	leaq	112(%rsp), %rcx
	movq	%r14, 48(%rsp)
	salq	$2, %rax
	movq	%rcx, 56(%rsp)
	movq	%rax, 88(%rsp)
	leaq	(%rcx,%rax), %rbx
	leaq	104(%rsp), %rax
	movq	%rax, 40(%rsp)
.L7:
	movss	(%r12,%r8,4), %xmm0
	movq	%r8, 8(%rsp)
	call	expf@PLT
	movq	88(%rsp), %rdx
	movq	56(%rsp), %rdi
	xorl	%esi, %esi
	movl	$16384, %ecx
	movss	%xmm0, 20(%rsp)
	call	__memset_chk@PLT
	movq	8(%rsp), %r8
	movq	%r15, 64(%rsp)
	movq	%r8, %r14
	movq	%r8, %rdx
	addq	$1, %r8
	salq	$7, %r14
	movq	%r8, 72(%rsp)
	movq	%rdx, 80(%rsp)
	leaq	128(%r14), %r13
	movq	%r14, %r15
	movq	%r12, %r14
	movq	56(%rsp), %r12
	.p2align 4,,10
	.p2align 3
.L3:
	movss	256(%r14,%r15), %xmm0
	call	expf@PLT
	movq	40(%rsp), %rsi
	movq	48(%rsp), %rdi
	movss	8448(%r14,%r15), %xmm2
	movaps	%xmm0, %xmm9
	movss	%xmm0, 36(%rsp)
	movss	20(%rsp), %xmm0
	xorps	.LC2(%rip), %xmm9
	mulss	%xmm2, %xmm0
	movss	%xmm2, 32(%rsp)
	movss	%xmm9, 8(%rsp)
	call	sincosf@PLT
	movss	20(%rsp), %xmm0
	mulss	8(%rsp), %xmm0
	movss	104(%rsp), %xmm4
	movss	108(%rsp), %xmm6
	movss	%xmm4, 28(%rsp)
	movss	%xmm6, 24(%rsp)
	call	expf@PLT
	movss	28(%rsp), %xmm4
	movq	%r12, %rax
	movss	32(%rsp), %xmm2
	movss	24(%rsp), %xmm3
	movss	36(%rsp), %xmm1
	mulss	%xmm0, %xmm4
	movss	8(%rsp), %xmm9
	movss	16640(%r14,%r15,2), %xmm6
	movss	16644(%r14,%r15,2), %xmm8
	mulss	%xmm0, %xmm3
	movaps	%xmm2, %xmm0
	mulss	%xmm2, %xmm0
	movaps	%xmm9, %xmm5
	mulss	%xmm1, %xmm1
	movaps	%xmm4, %xmm7
	subss	.LC1(%rip), %xmm7
	addss	%xmm0, %xmm1
	mulss	%xmm7, %xmm5
	movaps	%xmm2, %xmm0
	mulss	%xmm3, %xmm0
	mulss	%xmm7, %xmm2
	movaps	%xmm6, %xmm7
	addss	%xmm0, %xmm5
	movaps	%xmm9, %xmm0
	mulss	%xmm3, %xmm0
	divss	%xmm1, %xmm5
	subss	%xmm2, %xmm0
	movss	.LC1(%rip), %xmm2
	divss	%xmm1, %xmm0
	movaps	%xmm8, %xmm1
	mulss	%xmm5, %xmm7
	mulss	%xmm5, %xmm8
	mulss	%xmm0, %xmm1
	mulss	%xmm0, %xmm6
	subss	%xmm1, %xmm7
	pxor	%xmm1, %xmm1
	addss	%xmm8, %xmm6
	testl	%ebp, %ebp
	jle	.L5
	.p2align 4,,10
	.p2align 3
.L2:
	movaps	%xmm7, %xmm0
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
	cmpq	%rax, %rbx
	jne	.L2
.L5:
	addq	$4, %r15
	cmpq	%r13, %r15
	jne	.L3
	movq	80(%rsp), %rdx
	movq	64(%rsp), %r15
	movq	%r14, %r12
	movq	72(%rsp), %r8
	movss	33024(%r14,%rdx,4), %xmm2
	testl	%ebp, %ebp
	jle	.L11
	movq	96(%rsp), %rax
	leaq	256(%r15), %rcx
	xorl	%edi, %edi
	xorl	%esi, %esi
	leaq	(%rax,%rdx,4), %r13
	.p2align 4,,10
	.p2align 3
.L10:
	movss	(%rcx), %xmm1
	movq	%rcx, %rax
	movq	%rbx, %rdx
	mulss	%xmm2, %xmm1
	.p2align 4,,10
	.p2align 3
.L9:
	movss	-4(%rdx), %xmm0
	mulss	(%rax), %xmm0
	subq	$256, %rax
	subq	$4, %rdx
	addss	%xmm0, %xmm1
	cmpq	%rax, %r15
	jne	.L9
	addl	$1, %esi
	movss	%xmm1, 0(%r13,%rdi)
	addq	$256, %rcx
	addq	$256, %rdi
	cmpl	%esi, %ebp
	jne	.L10
.L11:
	addq	$4, %r15
	cmpq	$64, %r8
	jne	.L7
	movq	16504(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L26
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
.L26:
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
