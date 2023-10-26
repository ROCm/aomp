	.text
	.file	"flags.c"
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movl	$0, -4(%rbp)
	movl	nt, %eax
	movl	%eax, -8(%rbp)
	movl	-8(%rbp), %eax
	movl	%eax, -16(%rbp)
	movq	-16(%rbp), %rsi
	movabsq	$arr, %rdi
	callq	__omp_offloading_801_5fc4cb1_main_l7
	movl	$0, -20(%rbp)
	movl	$0, -24(%rbp)
.LBB0_1:                                # %for.cond
                                        # =>This Inner Loop Header: Depth=1
	cmpl	$100, -24(%rbp)
	jge	.LBB0_6
# %bb.2:                                # %for.body
                                        #   in Loop: Header=BB0_1 Depth=1
	movslq	-24(%rbp), %rax
	movl	arr(,%rax,4), %eax
	cmpl	-24(%rbp), %eax
	je	.LBB0_4
# %bb.3:                                # %if.then
                                        #   in Loop: Header=BB0_1 Depth=1
	movl	-20(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -20(%rbp)
.LBB0_4:                                # %if.end
                                        #   in Loop: Header=BB0_1 Depth=1
	jmp	.LBB0_5
.LBB0_5:                                # %for.inc
                                        #   in Loop: Header=BB0_1 Depth=1
	movl	-24(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -24(%rbp)
	jmp	.LBB0_1
.LBB0_6:                                # %for.end
	cmpl	$0, -20(%rbp)
	jne	.LBB0_8
# %bb.7:                                # %if.then3
	movq	stderr, %rdi
	movabsq	$.L.str, %rsi
	movb	$0, %al
	callq	fprintf
	movl	$0, -4(%rbp)
	jmp	.LBB0_9
.LBB0_8:                                # %if.else
	movq	stderr, %rdi
	movl	-20(%rbp), %edx
	movabsq	$.L.str.2, %rsi
	movb	$0, %al
	callq	fprintf
	movl	$1, -4(%rbp)
.LBB0_9:                                # %return
	movl	-4(%rbp), %eax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function __omp_offloading_801_5fc4cb1_main_l7
	.type	__omp_offloading_801_5fc4cb1_main_l7,@function
__omp_offloading_801_5fc4cb1_main_l7:   # @__omp_offloading_801_5fc4cb1_main_l7
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rcx
	movl	-16(%rbp), %eax
	movl	%eax, -24(%rbp)
	movq	-24(%rbp), %r8
	movabsq	$.L__unnamed_1, %rdi
	movl	$2, %esi
	movabsq	$.omp_outlined., %rdx
	movb	$0, %al
	callq	__kmpc_fork_teams@PLT
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end1:
	.size	__omp_offloading_801_5fc4cb1_main_l7, .Lfunc_end1-__omp_offloading_801_5fc4cb1_main_l7
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function .omp_outlined.
	.type	.omp_outlined.,@function
.omp_outlined.:                         # @.omp_outlined.
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$112, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	%rcx, -32(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -72(%rbp)                 # 8-byte Spill
	movl	$0, -44(%rbp)
	movl	$99, -48(%rbp)
	movl	$1, -52(%rbp)
	movl	$0, -56(%rbp)
	movq	-8(%rbp), %rax
	movl	(%rax), %esi
	movl	%esi, -64(%rbp)                 # 4-byte Spill
	movabsq	$.L__unnamed_2, %rdi
	movl	$92, %edx
	leaq	-56(%rbp), %rcx
	leaq	-44(%rbp), %r8
	leaq	-48(%rbp), %r9
	leaq	-52(%rbp), %rax
	movq	%rax, (%rsp)
	movl	$1, 8(%rsp)
	movl	$1, 16(%rsp)
	callq	__kmpc_for_static_init_4
	cmpl	$99, -48(%rbp)
	jle	.LBB2_2
# %bb.1:                                # %cond.true
	movl	$99, %eax
	movl	%eax, -76(%rbp)                 # 4-byte Spill
	jmp	.LBB2_3
.LBB2_2:                                # %cond.false
	movl	-48(%rbp), %eax
	movl	%eax, -76(%rbp)                 # 4-byte Spill
.LBB2_3:                                # %cond.end
	movl	-76(%rbp), %eax                 # 4-byte Reload
	movl	%eax, -48(%rbp)
	movl	-44(%rbp), %eax
	movl	%eax, -36(%rbp)
.LBB2_4:                                # %omp.inner.for.cond
                                        # =>This Inner Loop Header: Depth=1
	movl	-36(%rbp), %eax
	cmpl	-48(%rbp), %eax
	jg	.LBB2_7
# %bb.5:                                # %omp.inner.for.body
                                        #   in Loop: Header=BB2_4 Depth=1
	movl	-64(%rbp), %esi                 # 4-byte Reload
	movl	-32(%rbp), %edx
	movabsq	$.L__unnamed_1, %rdi
	callq	__kmpc_push_num_threads@PLT
	movq	-72(%rbp), %r9                  # 8-byte Reload
	movl	-44(%rbp), %eax
	movl	%eax, %ecx
	movl	-48(%rbp), %eax
	movl	%eax, %r8d
	movabsq	$.L__unnamed_1, %rdi
	movl	$3, %esi
	movabsq	$.omp_outlined..1, %rdx
	movb	$0, %al
	callq	__kmpc_fork_call@PLT
# %bb.6:                                # %omp.inner.for.inc
                                        #   in Loop: Header=BB2_4 Depth=1
	movl	-36(%rbp), %eax
	addl	-52(%rbp), %eax
	movl	%eax, -36(%rbp)
	jmp	.LBB2_4
.LBB2_7:                                # %omp.inner.for.end
	jmp	.LBB2_8
.LBB2_8:                                # %omp.loop.exit
	movl	-64(%rbp), %esi                 # 4-byte Reload
	movabsq	$.L__unnamed_2, %rdi
	callq	__kmpc_for_static_fini@PLT
	addq	$112, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end2:
	.size	.omp_outlined., .Lfunc_end2-.omp_outlined.
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function .omp_outlined..1
	.type	.omp_outlined..1,@function
.omp_outlined..1:                       # @.omp_outlined..1
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$112, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	%rcx, -32(%rbp)
	movq	%r8, -40(%rbp)
	movq	-40(%rbp), %rax
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	movl	$0, -52(%rbp)
	movl	$99, -56(%rbp)
	movq	-24(%rbp), %rax
	movl	%eax, %ecx
	movq	-32(%rbp), %rax
                                        # kill: def $eax killed $eax killed $rax
	movl	%ecx, -52(%rbp)
	movl	%eax, -56(%rbp)
	movl	$1, -60(%rbp)
	movl	$0, -64(%rbp)
	movq	-8(%rbp), %rax
	movl	(%rax), %esi
	movl	%esi, -72(%rbp)                 # 4-byte Spill
	movabsq	$.L__unnamed_3, %rdi
	movl	$34, %edx
	leaq	-64(%rbp), %rcx
	leaq	-52(%rbp), %r8
	leaq	-56(%rbp), %r9
	leaq	-60(%rbp), %rax
	movq	%rax, (%rsp)
	movl	$1, 8(%rsp)
	movl	$1, 16(%rsp)
	callq	__kmpc_for_static_init_4
	cmpl	$99, -56(%rbp)
	jle	.LBB3_2
# %bb.1:                                # %cond.true
	movl	$99, %eax
	movl	%eax, -84(%rbp)                 # 4-byte Spill
	jmp	.LBB3_3
.LBB3_2:                                # %cond.false
	movl	-56(%rbp), %eax
	movl	%eax, -84(%rbp)                 # 4-byte Spill
.LBB3_3:                                # %cond.end
	movl	-84(%rbp), %eax                 # 4-byte Reload
	movl	%eax, -56(%rbp)
	movl	-52(%rbp), %eax
	movl	%eax, -44(%rbp)
.LBB3_4:                                # %omp.inner.for.cond
                                        # =>This Inner Loop Header: Depth=1
	movl	-44(%rbp), %eax
	cmpl	-56(%rbp), %eax
	jg	.LBB3_8
# %bb.5:                                # %omp.inner.for.body
                                        #   in Loop: Header=BB3_4 Depth=1
	movq	-80(%rbp), %rax                 # 8-byte Reload
	movl	-44(%rbp), %ecx
	shll	$0, %ecx
	addl	$0, %ecx
	movl	%ecx, -68(%rbp)
	movl	-68(%rbp), %edx
	movslq	-68(%rbp), %rcx
	movl	%edx, (%rax,%rcx,4)
# %bb.6:                                # %omp.body.continue
                                        #   in Loop: Header=BB3_4 Depth=1
	jmp	.LBB3_7
.LBB3_7:                                # %omp.inner.for.inc
                                        #   in Loop: Header=BB3_4 Depth=1
	movl	-44(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -44(%rbp)
	jmp	.LBB3_4
.LBB3_8:                                # %omp.inner.for.end
	jmp	.LBB3_9
.LBB3_9:                                # %omp.loop.exit
	movl	-72(%rbp), %esi                 # 4-byte Reload
	movabsq	$.L__unnamed_2, %rdi
	callq	__kmpc_for_static_fini@PLT
	addq	$112, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end3:
	.size	.omp_outlined..1, .Lfunc_end3-.omp_outlined..1
	.cfi_endproc
                                        # -- End function
	.type	arr,@object                     # @arr
	.bss
	.globl	arr
	.p2align	4, 0x0
arr:
	.zero	400
	.size	arr, 400

	.type	nt,@object                      # @nt
	.data
	.globl	nt
	.p2align	2, 0x0
nt:
	.long	12                              # 0xc
	.size	nt, 4

	.type	.L__unnamed_4,@object           # @0
	.section	.rodata.str1.1,"aMS",@progbits,1
.L__unnamed_4:
	.asciz	";unknown;unknown;0;0;;"
	.size	.L__unnamed_4, 23

	.type	.L__unnamed_2,@object           # @1
	.section	.rodata,"a",@progbits
	.p2align	3, 0x0
.L__unnamed_2:
	.long	0                               # 0x0
	.long	2050                            # 0x802
	.long	0                               # 0x0
	.long	22                              # 0x16
	.quad	.L__unnamed_4
	.size	.L__unnamed_2, 24

	.type	.L__unnamed_3,@object           # @2
	.p2align	3, 0x0
.L__unnamed_3:
	.long	0                               # 0x0
	.long	514                             # 0x202
	.long	0                               # 0x0
	.long	22                              # 0x16
	.quad	.L__unnamed_4
	.size	.L__unnamed_3, 24

	.type	.L__unnamed_1,@object           # @3
	.p2align	3, 0x0
.L__unnamed_1:
	.long	0                               # 0x0
	.long	2                               # 0x2
	.long	0                               # 0x0
	.long	22                              # 0x16
	.quad	.L__unnamed_4
	.size	.L__unnamed_1, 24

	.type	.L.str,@object                  # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"Success\n"
	.size	.L.str, 9

	.type	.L.str.2,@object                # @.str.2
.L.str.2:
	.asciz	"Failed\nErrors: %d\n"
	.size	.L.str.2, 19

	.ident	"AMD clang version 16.0.0 (  23102 461658c3abfc5f55172edb0382bd6aad66559d7b)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __omp_offloading_801_5fc4cb1_main_l7
	.addrsig_sym .omp_outlined.
	.addrsig_sym __kmpc_for_static_init_4
	.addrsig_sym .omp_outlined..1
	.addrsig_sym __kmpc_for_static_fini
	.addrsig_sym __kmpc_push_num_threads
	.addrsig_sym __kmpc_fork_call
	.addrsig_sym __kmpc_fork_teams
	.addrsig_sym fprintf
	.addrsig_sym arr
	.addrsig_sym nt
	.addrsig_sym stderr
