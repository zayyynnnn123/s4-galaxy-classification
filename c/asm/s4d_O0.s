	.file	"s4d.c"
	.text
	.type	complex_mul, @function
complex_mul:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movss	%xmm0, -4(%rbp)
	movss	%xmm1, -8(%rbp)
	movss	%xmm2, -12(%rbp)
	movss	%xmm3, -16(%rbp)
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movss	-4(%rbp), %xmm0
	mulss	-12(%rbp), %xmm0
	movss	-8(%rbp), %xmm1
	mulss	-16(%rbp), %xmm1
	subss	%xmm1, %xmm0
	movq	-24(%rbp), %rax
	movss	%xmm0, (%rax)
	movss	-4(%rbp), %xmm0
	movaps	%xmm0, %xmm1
	mulss	-16(%rbp), %xmm1
	movss	-8(%rbp), %xmm0
	mulss	-12(%rbp), %xmm0
	addss	%xmm1, %xmm0
	movq	-32(%rbp), %rax
	movss	%xmm0, (%rax)
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	complex_mul, .-complex_mul
	.type	complex_exp, @function
complex_exp:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movss	%xmm0, -20(%rbp)
	movss	%xmm1, -24(%rbp)
	movq	%rdi, -32(%rbp)
	movq	%rsi, -40(%rbp)
	movl	-20(%rbp), %eax
	movd	%eax, %xmm0
	call	expf@PLT
	movd	%xmm0, %eax
	movl	%eax, -4(%rbp)
	movl	-24(%rbp), %eax
	movd	%eax, %xmm0
	call	cosf@PLT
	movd	%xmm0, %eax
	movd	%eax, %xmm0
	mulss	-4(%rbp), %xmm0
	movq	-32(%rbp), %rax
	movss	%xmm0, (%rax)
	movl	-24(%rbp), %eax
	movd	%eax, %xmm0
	call	sinf@PLT
	movd	%xmm0, %eax
	movd	%eax, %xmm0
	mulss	-4(%rbp), %xmm0
	movq	-40(%rbp), %rax
	movss	%xmm0, (%rax)
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	complex_exp, .-complex_exp
	.type	complex_div, @function
complex_div:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movss	%xmm0, -20(%rbp)
	movss	%xmm1, -24(%rbp)
	movss	%xmm2, -28(%rbp)
	movss	%xmm3, -32(%rbp)
	movq	%rdi, -40(%rbp)
	movq	%rsi, -48(%rbp)
	movss	-28(%rbp), %xmm0
	movaps	%xmm0, %xmm1
	mulss	%xmm0, %xmm1
	movss	-32(%rbp), %xmm0
	mulss	%xmm0, %xmm0
	addss	%xmm1, %xmm0
	movss	%xmm0, -4(%rbp)
	movss	-20(%rbp), %xmm0
	movaps	%xmm0, %xmm1
	mulss	-28(%rbp), %xmm1
	movss	-24(%rbp), %xmm0
	mulss	-32(%rbp), %xmm0
	addss	%xmm1, %xmm0
	divss	-4(%rbp), %xmm0
	movq	-40(%rbp), %rax
	movss	%xmm0, (%rax)
	movss	-24(%rbp), %xmm0
	mulss	-28(%rbp), %xmm0
	movss	-20(%rbp), %xmm1
	mulss	-32(%rbp), %xmm1
	subss	%xmm1, %xmm0
	divss	-4(%rbp), %xmm0
	movq	-48(%rbp), %rax
	movss	%xmm0, (%rax)
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	complex_div, .-complex_div
	.globl	s4d_forward
	.type	s4d_forward, @function
s4d_forward:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	leaq	-16384(%rsp), %r11
.LPSRL0:
	subq	$4096, %rsp
	orq	$0, (%rsp)
	cmpq	%r11, %rsp
	jne	.LPSRL0
	subq	$160, %rsp
	movq	%rdi, -16520(%rbp)
	movq	%rsi, -16528(%rbp)
	movq	%rdx, -16536(%rbp)
	movl	%ecx, -16540(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movl	$0, -16468(%rbp)
	jmp	.L5
.L14:
	movq	-16536(%rbp), %rax
	movl	-16468(%rbp), %edx
	movslq	%edx, %rdx
	movl	(%rax,%rdx,4), %eax
	movd	%eax, %xmm0
	call	expf@PLT
	movd	%xmm0, %eax
	movl	%eax, -16440(%rbp)
	movl	-16540(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	-16400(%rbp), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	memset@PLT
	movl	$0, -16464(%rbp)
	jmp	.L6
.L9:
	movl	-16468(%rbp), %eax
	sall	$5, %eax
	movl	%eax, %edx
	movl	-16464(%rbp), %eax
	addl	%eax, %edx
	movq	-16536(%rbp), %rax
	movslq	%edx, %rdx
	addq	$64, %rdx
	movl	(%rax,%rdx,4), %eax
	movd	%eax, %xmm0
	call	expf@PLT
	movd	%xmm0, %eax
	movss	.LC0(%rip), %xmm0
	movd	%eax, %xmm4
	xorps	%xmm0, %xmm4
	movaps	%xmm4, %xmm0
	movss	%xmm0, -16432(%rbp)
	movl	-16468(%rbp), %eax
	sall	$5, %eax
	movl	%eax, %edx
	movl	-16464(%rbp), %eax
	addl	%eax, %edx
	movq	-16536(%rbp), %rax
	movslq	%edx, %rdx
	addq	$2112, %rdx
	movss	(%rax,%rdx,4), %xmm0
	movss	%xmm0, -16428(%rbp)
	movss	-16440(%rbp), %xmm0
	mulss	-16432(%rbp), %xmm0
	movss	%xmm0, -16424(%rbp)
	movss	-16440(%rbp), %xmm0
	mulss	-16428(%rbp), %xmm0
	movss	%xmm0, -16420(%rbp)
	leaq	-16496(%rbp), %rcx
	leaq	-16500(%rbp), %rdx
	movss	-16420(%rbp), %xmm0
	movl	-16424(%rbp), %eax
	movq	%rcx, %rsi
	movq	%rdx, %rdi
	movaps	%xmm0, %xmm1
	movd	%eax, %xmm0
	call	complex_exp
	movss	-16500(%rbp), %xmm0
	movss	.LC1(%rip), %xmm1
	subss	%xmm1, %xmm0
	movss	%xmm0, -16416(%rbp)
	movss	-16496(%rbp), %xmm0
	movss	%xmm0, -16412(%rbp)
	movl	-16468(%rbp), %eax
	sall	$5, %eax
	movl	%eax, %edx
	movl	-16464(%rbp), %eax
	addl	%edx, %eax
	leal	(%rax,%rax), %edx
	movq	-16536(%rbp), %rax
	movslq	%edx, %rdx
	addq	$4160, %rdx
	movss	(%rax,%rdx,4), %xmm0
	movss	%xmm0, -16408(%rbp)
	movl	-16468(%rbp), %eax
	sall	$5, %eax
	movl	%eax, %edx
	movl	-16464(%rbp), %eax
	addl	%edx, %eax
	addl	%eax, %eax
	leal	1(%rax), %edx
	movq	-16536(%rbp), %rax
	movslq	%edx, %rdx
	addq	$4160, %rdx
	movss	(%rax,%rdx,4), %xmm0
	movss	%xmm0, -16404(%rbp)
	leaq	-16488(%rbp), %rcx
	leaq	-16492(%rbp), %rdx
	movss	-16428(%rbp), %xmm2
	movss	-16432(%rbp), %xmm1
	movss	-16412(%rbp), %xmm0
	movl	-16416(%rbp), %eax
	movq	%rcx, %rsi
	movq	%rdx, %rdi
	movaps	%xmm2, %xmm3
	movaps	%xmm1, %xmm2
	movaps	%xmm0, %xmm1
	movd	%eax, %xmm0
	call	complex_div
	movss	-16488(%rbp), %xmm2
	movss	-16492(%rbp), %xmm1
	leaq	-16480(%rbp), %rcx
	leaq	-16484(%rbp), %rdx
	movss	-16404(%rbp), %xmm0
	movl	-16408(%rbp), %eax
	movq	%rcx, %rsi
	movq	%rdx, %rdi
	movaps	%xmm2, %xmm3
	movaps	%xmm1, %xmm2
	movaps	%xmm0, %xmm1
	movd	%eax, %xmm0
	call	complex_mul
	movss	.LC1(%rip), %xmm0
	movss	%xmm0, -16452(%rbp)
	pxor	%xmm0, %xmm0
	movss	%xmm0, -16448(%rbp)
	movl	$0, -16460(%rbp)
	jmp	.L7
.L8:
	movl	-16460(%rbp), %eax
	cltq
	movss	-16400(%rbp,%rax,4), %xmm2
	movss	-16484(%rbp), %xmm0
	mulss	-16452(%rbp), %xmm0
	movss	-16480(%rbp), %xmm1
	mulss	-16448(%rbp), %xmm1
	subss	%xmm1, %xmm0
	addss	%xmm0, %xmm0
	addss	%xmm2, %xmm0
	movl	-16460(%rbp), %eax
	cltq
	movss	%xmm0, -16400(%rbp,%rax,4)
	movss	-16496(%rbp), %xmm2
	movss	-16500(%rbp), %xmm1
	leaq	-16472(%rbp), %rcx
	leaq	-16476(%rbp), %rdx
	movss	-16448(%rbp), %xmm0
	movl	-16452(%rbp), %eax
	movq	%rcx, %rsi
	movq	%rdx, %rdi
	movaps	%xmm2, %xmm3
	movaps	%xmm1, %xmm2
	movaps	%xmm0, %xmm1
	movd	%eax, %xmm0
	call	complex_mul
	movss	-16476(%rbp), %xmm0
	movss	%xmm0, -16452(%rbp)
	movss	-16472(%rbp), %xmm0
	movss	%xmm0, -16448(%rbp)
	addl	$1, -16460(%rbp)
.L7:
	movl	-16460(%rbp), %eax
	cmpl	-16540(%rbp), %eax
	jl	.L8
	addl	$1, -16464(%rbp)
.L6:
	cmpl	$31, -16464(%rbp)
	jle	.L9
	movq	-16536(%rbp), %rax
	movl	-16468(%rbp), %edx
	movslq	%edx, %rdx
	addq	$8256, %rdx
	movss	(%rax,%rdx,4), %xmm0
	movss	%xmm0, -16436(%rbp)
	movl	$0, -16460(%rbp)
	jmp	.L10
.L13:
	movl	-16460(%rbp), %eax
	sall	$6, %eax
	movl	%eax, %edx
	movl	-16468(%rbp), %eax
	addl	%edx, %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-16520(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	movss	-16436(%rbp), %xmm1
	mulss	%xmm1, %xmm0
	movss	%xmm0, -16444(%rbp)
	movl	$0, -16456(%rbp)
	jmp	.L11
.L12:
	movl	-16540(%rbp), %eax
	subl	$1, %eax
	subl	-16456(%rbp), %eax
	cltq
	movss	-16400(%rbp,%rax,4), %xmm1
	movl	-16460(%rbp), %eax
	subl	-16456(%rbp), %eax
	sall	$6, %eax
	movl	%eax, %edx
	movl	-16468(%rbp), %eax
	addl	%edx, %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-16520(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	mulss	%xmm1, %xmm0
	movss	-16444(%rbp), %xmm1
	addss	%xmm1, %xmm0
	movss	%xmm0, -16444(%rbp)
	addl	$1, -16456(%rbp)
.L11:
	movl	-16456(%rbp), %eax
	cmpl	-16460(%rbp), %eax
	jle	.L12
	movl	-16460(%rbp), %eax
	sall	$6, %eax
	movl	%eax, %edx
	movl	-16468(%rbp), %eax
	addl	%edx, %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-16528(%rbp), %rax
	addq	%rdx, %rax
	movss	-16444(%rbp), %xmm0
	movss	%xmm0, (%rax)
	addl	$1, -16460(%rbp)
.L10:
	movl	-16460(%rbp), %eax
	cmpl	-16540(%rbp), %eax
	jl	.L13
	addl	$1, -16468(%rbp)
.L5:
	cmpl	$63, -16468(%rbp)
	jle	.L14
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L15
	call	__stack_chk_fail@PLT
.L15:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	s4d_forward, .-s4d_forward
	.section	.rodata
	.align 16
.LC0:
	.long	-2147483648
	.long	0
	.long	0
	.long	0
	.align 4
.LC1:
	.long	1065353216
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
