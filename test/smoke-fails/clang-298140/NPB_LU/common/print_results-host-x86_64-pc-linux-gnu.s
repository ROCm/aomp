	.text
	.file	"print_results.c"
	.file	1 "/home/dteixeir/NASA/FT/common" "../common/type.h"
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3                               # -- Begin function print_results
.LCPI0_0:
	.quad	0x4000000000000000              # double 2
	.text
	.globl	print_results
	.p2align	4, 0x90
	.type	print_results,@function
print_results:                          # @print_results
.Lfunc_begin0:
	.file	2 "/home/dteixeir/NASA/FT/common" "print_results.c"
	.loc	2 10 0                          # print_results.c:10:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$80, %rsp
	movb	%sil, %al
	movq	96(%rbp), %rsi
	movq	88(%rbp), %rsi
	movq	80(%rbp), %rsi
	movq	72(%rbp), %rsi
	movq	64(%rbp), %rsi
	movq	56(%rbp), %rsi
	movq	48(%rbp), %rsi
	movq	40(%rbp), %rsi
	movq	32(%rbp), %rsi
	movl	24(%rbp), %esi
	movq	16(%rbp), %rsi
	movq	%rdi, -8(%rbp)
	movb	%al, -9(%rbp)
	movl	%edx, -16(%rbp)
	movl	%ecx, -20(%rbp)
	movl	%r8d, -24(%rbp)
	movl	%r9d, -28(%rbp)
	movsd	%xmm0, -40(%rbp)
	movsd	%xmm1, -48(%rbp)
.Ltmp0:
	.loc	2 14 45 prologue_end            # print_results.c:14:45
	movq	-8(%rbp), %rsi
	.loc	2 14 3 is_stmt 0                # print_results.c:14:3
	movabsq	$.L.str, %rdi
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	callq	printf
	.loc	2 15 52 is_stmt 1               # print_results.c:15:52
	movsbl	-9(%rbp), %esi
	.loc	2 15 3 is_stmt 0                # print_results.c:15:3
	movabsq	$.L.str.1, %rdi
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	callq	printf
.Ltmp1:
	.loc	2 22 13 is_stmt 1               # print_results.c:22:13
	cmpl	$0, -20(%rbp)
	.loc	2 22 20 is_stmt 0               # print_results.c:22:20
	jne	.LBB0_9
# %bb.1:
	.loc	2 22 28                         # print_results.c:22:28
	cmpl	$0, -24(%rbp)
.Ltmp2:
	.loc	2 22 8                          # print_results.c:22:8
	jne	.LBB0_9
# %bb.2:
.Ltmp3:
	.loc	2 23 12 is_stmt 1               # print_results.c:23:12
	movq	-8(%rbp), %rax
	movsbl	(%rax), %eax
	.loc	2 23 20 is_stmt 0               # print_results.c:23:20
	cmpl	$69, %eax
	.loc	2 23 29                         # print_results.c:23:29
	jne	.LBB0_7
# %bb.3:
	.loc	2 23 34                         # print_results.c:23:34
	movq	-8(%rbp), %rax
	movsbl	1(%rax), %eax
	.loc	2 23 42                         # print_results.c:23:42
	cmpl	$80, %eax
.Ltmp4:
	.loc	2 23 10                         # print_results.c:23:10
	jne	.LBB0_7
# %bb.4:
	.loc	2 0 10                          # print_results.c:0:10
	movl	$2, %eax
	cvtsi2sd	%rax, %xmm0
	leaq	-64(%rbp), %rax
	movq	%rax, -80(%rbp)                 # 8-byte Spill
.Ltmp5:
	.loc	2 24 42 is_stmt 1               # print_results.c:24:42
	cvtsi2sdl	-16(%rbp), %xmm1
	movabsq	$.LCPI0_0, %rax
	movsd	(%rax), %xmm0                   # xmm0 = mem[0],zero
	.loc	2 24 33 is_stmt 0               # print_results.c:24:33
	callq	pow
	movq	-80(%rbp), %rdi                 # 8-byte Reload
	.loc	2 24 7                          # print_results.c:24:7
	movabsq	$.L.str.2, %rsi
	movb	$1, %al
	callq	sprintf
	.loc	2 25 9 is_stmt 1                # print_results.c:25:9
	movl	$14, -68(%rbp)
.Ltmp6:
	.loc	2 26 12                         # print_results.c:26:12
	movslq	-68(%rbp), %rax
	movsbl	-64(%rbp,%rax), %eax
	.loc	2 26 20 is_stmt 0               # print_results.c:26:20
	cmpl	$46, %eax
.Ltmp7:
	.loc	2 26 12                         # print_results.c:26:12
	jne	.LBB0_6
# %bb.5:
.Ltmp8:
	.loc	2 27 9 is_stmt 1                # print_results.c:27:9
	movslq	-68(%rbp), %rax
	.loc	2 27 17 is_stmt 0               # print_results.c:27:17
	movb	$32, -64(%rbp,%rax)
	.loc	2 28 10 is_stmt 1               # print_results.c:28:10
	movl	-68(%rbp), %eax
	addl	$-1, %eax
	movl	%eax, -68(%rbp)
.Ltmp9:
.LBB0_6:
	.loc	2 0 10 is_stmt 0                # print_results.c:0:10
	leaq	-64(%rbp), %rsi
	.loc	2 30 12 is_stmt 1               # print_results.c:30:12
	movl	-68(%rbp), %eax
	.loc	2 30 13 is_stmt 0               # print_results.c:30:13
	addl	$1, %eax
	.loc	2 30 7                          # print_results.c:30:7
	cltq
	.loc	2 30 17                         # print_results.c:30:17
	movb	$0, -64(%rbp,%rax)
	.loc	2 31 7 is_stmt 1                # print_results.c:31:7
	movabsq	$.L.str.3, %rdi
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	callq	printf
	.loc	2 32 5                          # print_results.c:32:5
	jmp	.LBB0_8
.Ltmp10:
.LBB0_7:
	.loc	2 33 56                         # print_results.c:33:56
	movl	-16(%rbp), %esi
	.loc	2 33 7 is_stmt 0                # print_results.c:33:7
	movabsq	$.L.str.4, %rdi
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	callq	printf
.Ltmp11:
.LBB0_8:
	.loc	2 35 3 is_stmt 1                # print_results.c:35:3
	jmp	.LBB0_10
.Ltmp12:
.LBB0_9:
	.loc	2 36 59                         # print_results.c:36:59
	movl	-16(%rbp), %esi
	.loc	2 36 63 is_stmt 0               # print_results.c:36:63
	movl	-20(%rbp), %edx
	.loc	2 36 67                         # print_results.c:36:67
	movl	-24(%rbp), %ecx
	.loc	2 36 5                          # print_results.c:36:5
	movabsq	$.L.str.5, %rdi
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	callq	printf
.Ltmp13:
.LBB0_10:
	.loc	2 39 52 is_stmt 1               # print_results.c:39:52
	movl	-28(%rbp), %esi
	.loc	2 39 3 is_stmt 0                # print_results.c:39:3
	movabsq	$.L.str.6, %rdi
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	callq	printf
	.loc	2 40 55 is_stmt 1               # print_results.c:40:55
	movsd	-40(%rbp), %xmm0                # xmm0 = mem[0],zero
	.loc	2 40 3 is_stmt 0                # print_results.c:40:3
	movabsq	$.L.str.7, %rdi
	movb	$1, %al
	callq	printf
	.loc	2 41 52 is_stmt 1               # print_results.c:41:52
	movsd	-48(%rbp), %xmm0                # xmm0 = mem[0],zero
	.loc	2 41 3 is_stmt 0                # print_results.c:41:3
	movabsq	$.L.str.8, %rdi
	movb	$1, %al
	callq	printf
	.loc	2 42 40 is_stmt 1               # print_results.c:42:40
	movq	16(%rbp), %rsi
	.loc	2 42 3 is_stmt 0                # print_results.c:42:3
	movabsq	$.L.str.9, %rdi
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	callq	printf
.Ltmp14:
	.loc	2 43 8 is_stmt 1                # print_results.c:43:8
	cmpl	$0, 24(%rbp)
.Ltmp15:
	.loc	2 43 8 is_stmt 0                # print_results.c:43:8
	je	.LBB0_12
# %bb.11:
.Ltmp16:
	.loc	2 44 5 is_stmt 1                # print_results.c:44:5
	movabsq	$.L.str.10, %rdi
	movabsq	$.L.str.11, %rsi
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	callq	printf
	jmp	.LBB0_13
.LBB0_12:
	.loc	2 46 5                          # print_results.c:46:5
	movabsq	$.L.str.10, %rdi
	movabsq	$.L.str.12, %rsi
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	callq	printf
.Ltmp17:
.LBB0_13:
	.loc	2 47 52                         # print_results.c:47:52
	movq	32(%rbp), %rsi
	.loc	2 47 3 is_stmt 0                # print_results.c:47:3
	movabsq	$.L.str.13, %rdi
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	callq	printf
	.loc	2 48 52 is_stmt 1               # print_results.c:48:52
	movq	40(%rbp), %rsi
	.loc	2 48 3 is_stmt 0                # print_results.c:48:3
	movabsq	$.L.str.14, %rdi
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	callq	printf
	.loc	2 51 38 is_stmt 1               # print_results.c:51:38
	movq	48(%rbp), %rsi
	.loc	2 50 3                          # print_results.c:50:3
	movabsq	$.L.str.15, %rdi
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	callq	printf
	.loc	2 52 38                         # print_results.c:52:38
	movq	56(%rbp), %rsi
	.loc	2 52 3 is_stmt 0                # print_results.c:52:3
	movabsq	$.L.str.16, %rdi
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	callq	printf
	.loc	2 53 38 is_stmt 1               # print_results.c:53:38
	movq	64(%rbp), %rsi
	.loc	2 53 3 is_stmt 0                # print_results.c:53:3
	movabsq	$.L.str.17, %rdi
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	callq	printf
	.loc	2 54 38 is_stmt 1               # print_results.c:54:38
	movq	72(%rbp), %rsi
	.loc	2 54 3 is_stmt 0                # print_results.c:54:3
	movabsq	$.L.str.18, %rdi
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	callq	printf
	.loc	2 55 38 is_stmt 1               # print_results.c:55:38
	movq	80(%rbp), %rsi
	.loc	2 55 3 is_stmt 0                # print_results.c:55:3
	movabsq	$.L.str.19, %rdi
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	callq	printf
	.loc	2 56 38 is_stmt 1               # print_results.c:56:38
	movq	88(%rbp), %rsi
	.loc	2 56 3 is_stmt 0                # print_results.c:56:3
	movabsq	$.L.str.20, %rdi
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	callq	printf
	.loc	2 57 38 is_stmt 1               # print_results.c:57:38
	movq	96(%rbp), %rsi
	.loc	2 57 3 is_stmt 0                # print_results.c:57:3
	movabsq	$.L.str.21, %rdi
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	callq	printf
	.loc	2 59 3 is_stmt 1                # print_results.c:59:3
	movabsq	$.L.str.22, %rdi
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	callq	printf
	.loc	2 65 1                          # print_results.c:65:1
	addq	$80, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp18:
.Lfunc_end0:
	.size	print_results, .Lfunc_end0-print_results
	.cfi_endproc
                                        # -- End function
	.type	.L.str,@object                  # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"\n\n %s Benchmark Completed.\n"
	.size	.L.str, 28

	.type	.L.str.1,@object                # @.str.1
.L.str.1:
	.asciz	" Class           =             %12c\n"
	.size	.L.str.1, 37

	.type	.L.str.2,@object                # @.str.2
.L.str.2:
	.asciz	"%15.0lf"
	.size	.L.str.2, 8

	.type	.L.str.3,@object                # @.str.3
.L.str.3:
	.asciz	" Size            =          %15s\n"
	.size	.L.str.3, 34

	.type	.L.str.4,@object                # @.str.4
.L.str.4:
	.asciz	" Size            =             %12d\n"
	.size	.L.str.4, 37

	.type	.L.str.5,@object                # @.str.5
.L.str.5:
	.asciz	" Size            =           %4dx%4dx%4d\n"
	.size	.L.str.5, 42

	.type	.L.str.6,@object                # @.str.6
.L.str.6:
	.asciz	" Iterations      =             %12d\n"
	.size	.L.str.6, 37

	.type	.L.str.7,@object                # @.str.7
.L.str.7:
	.asciz	" Time in seconds =             %12.2lf\n"
	.size	.L.str.7, 40

	.type	.L.str.8,@object                # @.str.8
.L.str.8:
	.asciz	" Mop/s total     =          %15.2lf\n"
	.size	.L.str.8, 37

	.type	.L.str.9,@object                # @.str.9
.L.str.9:
	.asciz	" Operation type  = %24s\n"
	.size	.L.str.9, 25

	.type	.L.str.10,@object               # @.str.10
.L.str.10:
	.asciz	" Verification    =             %12s\n"
	.size	.L.str.10, 37

	.type	.L.str.11,@object               # @.str.11
.L.str.11:
	.asciz	"SUCCESSFUL"
	.size	.L.str.11, 11

	.type	.L.str.12,@object               # @.str.12
.L.str.12:
	.asciz	"UNSUCCESSFUL"
	.size	.L.str.12, 13

	.type	.L.str.13,@object               # @.str.13
.L.str.13:
	.asciz	" Version         =             %12s\n"
	.size	.L.str.13, 37

	.type	.L.str.14,@object               # @.str.14
.L.str.14:
	.asciz	" Compile date    =             %12s\n"
	.size	.L.str.14, 37

	.type	.L.str.15,@object               # @.str.15
.L.str.15:
	.asciz	"\n Compile options:\n    CC           = %s\n"
	.size	.L.str.15, 42

	.type	.L.str.16,@object               # @.str.16
.L.str.16:
	.asciz	"    CLINK        = %s\n"
	.size	.L.str.16, 23

	.type	.L.str.17,@object               # @.str.17
.L.str.17:
	.asciz	"    C_LIB        = %s\n"
	.size	.L.str.17, 23

	.type	.L.str.18,@object               # @.str.18
.L.str.18:
	.asciz	"    C_INC        = %s\n"
	.size	.L.str.18, 23

	.type	.L.str.19,@object               # @.str.19
.L.str.19:
	.asciz	"    CFLAGS       = %s\n"
	.size	.L.str.19, 23

	.type	.L.str.20,@object               # @.str.20
.L.str.20:
	.asciz	"    CLINKFLAGS   = %s\n"
	.size	.L.str.20, 23

	.type	.L.str.21,@object               # @.str.21
.L.str.21:
	.asciz	"    RAND         = %s\n"
	.size	.L.str.21, 23

	.type	.L.str.22,@object               # @.str.22
.L.str.22:
	.asciz	"\n--------------------------------------\n Please send all errors/feedbacks to:\n Center for Manycore Programming\n cmp@aces.snu.ac.kr\n http://aces.snu.ac.kr\n--------------------------------------\n\n"
	.size	.L.str.22, 195

	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	37                              # DW_AT_producer
	.byte	14                              # DW_FORM_strp
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	14                              # DW_FORM_strp
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	4                               # DW_TAG_enumeration_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	40                              # DW_TAG_enumerator
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	28                              # DW_AT_const_value
	.byte	15                              # DW_FORM_udata
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.byte	36                              # DW_TAG_base_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	62                              # DW_AT_encoding
	.byte	11                              # DW_FORM_data1
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	39                              # DW_AT_prototyped
	.byte	25                              # DW_FORM_flag_present
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	6                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	7                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	8                               # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	9                               # Abbreviation Code
	.byte	22                              # DW_TAG_typedef
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	10                              # Abbreviation Code
	.byte	1                               # DW_TAG_array_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	11                              # Abbreviation Code
	.byte	33                              # DW_TAG_subrange_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	55                              # DW_AT_count
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	12                              # Abbreviation Code
	.byte	36                              # DW_TAG_base_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	62                              # DW_AT_encoding
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0x1b6 DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	12                              # DW_AT_language
	.long	.Linfo_string1                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string2                  # DW_AT_comp_dir
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	2                               # Abbrev [2] 0x2a:0x15 DW_TAG_enumeration_type
	.long	63                              # DW_AT_type
	.byte	4                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x32:0x6 DW_TAG_enumerator
	.long	.Linfo_string4                  # DW_AT_name
	.byte	0                               # DW_AT_const_value
	.byte	3                               # Abbrev [3] 0x38:0x6 DW_TAG_enumerator
	.long	.Linfo_string5                  # DW_AT_name
	.byte	1                               # DW_AT_const_value
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x3f:0x7 DW_TAG_base_type
	.long	.Linfo_string3                  # DW_AT_name
	.byte	7                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	5                               # Abbrev [5] 0x46:0x142 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string6                  # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	6                               # DW_AT_decl_line
                                        # DW_AT_prototyped
                                        # DW_AT_external
	.byte	6                               # Abbrev [6] 0x5b:0xe DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string7                  # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	6                               # DW_AT_decl_line
	.long	392                             # DW_AT_type
	.byte	6                               # Abbrev [6] 0x69:0xe DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	119
	.long	.Linfo_string9                  # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	6                               # DW_AT_decl_line
	.long	397                             # DW_AT_type
	.byte	6                               # Abbrev [6] 0x77:0xe DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string10                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	6                               # DW_AT_decl_line
	.long	404                             # DW_AT_type
	.byte	6                               # Abbrev [6] 0x85:0xe DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	108
	.long	.Linfo_string12                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	6                               # DW_AT_decl_line
	.long	404                             # DW_AT_type
	.byte	6                               # Abbrev [6] 0x93:0xe DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	104
	.long	.Linfo_string13                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	6                               # DW_AT_decl_line
	.long	404                             # DW_AT_type
	.byte	6                               # Abbrev [6] 0xa1:0xe DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	100
	.long	.Linfo_string14                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	6                               # DW_AT_decl_line
	.long	404                             # DW_AT_type
	.byte	6                               # Abbrev [6] 0xaf:0xe DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	88
	.long	.Linfo_string15                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.long	411                             # DW_AT_type
	.byte	6                               # Abbrev [6] 0xbd:0xe DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	80
	.long	.Linfo_string17                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.long	411                             # DW_AT_type
	.byte	6                               # Abbrev [6] 0xcb:0xe DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	16
	.long	.Linfo_string18                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.long	392                             # DW_AT_type
	.byte	6                               # Abbrev [6] 0xd9:0xe DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	24
	.long	.Linfo_string19                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.long	418                             # DW_AT_type
	.byte	6                               # Abbrev [6] 0xe7:0xe DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	32
	.long	.Linfo_string21                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.long	392                             # DW_AT_type
	.byte	6                               # Abbrev [6] 0xf5:0xe DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	40
	.long	.Linfo_string22                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	8                               # DW_AT_decl_line
	.long	392                             # DW_AT_type
	.byte	6                               # Abbrev [6] 0x103:0xe DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	48
	.long	.Linfo_string23                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	8                               # DW_AT_decl_line
	.long	392                             # DW_AT_type
	.byte	6                               # Abbrev [6] 0x111:0xe DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	56
	.long	.Linfo_string24                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	8                               # DW_AT_decl_line
	.long	392                             # DW_AT_type
	.byte	6                               # Abbrev [6] 0x11f:0xf DW_TAG_formal_parameter
	.byte	3                               # DW_AT_location
	.byte	145
	.asciz	"\300"
	.long	.Linfo_string25                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	8                               # DW_AT_decl_line
	.long	392                             # DW_AT_type
	.byte	6                               # Abbrev [6] 0x12e:0xf DW_TAG_formal_parameter
	.byte	3                               # DW_AT_location
	.byte	145
	.asciz	"\310"
	.long	.Linfo_string26                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	8                               # DW_AT_decl_line
	.long	392                             # DW_AT_type
	.byte	6                               # Abbrev [6] 0x13d:0xf DW_TAG_formal_parameter
	.byte	3                               # DW_AT_location
	.byte	145
	.asciz	"\320"
	.long	.Linfo_string27                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	8                               # DW_AT_decl_line
	.long	392                             # DW_AT_type
	.byte	6                               # Abbrev [6] 0x14c:0xf DW_TAG_formal_parameter
	.byte	3                               # DW_AT_location
	.byte	145
	.asciz	"\330"
	.long	.Linfo_string28                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	9                               # DW_AT_decl_line
	.long	392                             # DW_AT_type
	.byte	6                               # Abbrev [6] 0x15b:0xf DW_TAG_formal_parameter
	.byte	3                               # DW_AT_location
	.byte	145
	.asciz	"\340"
	.long	.Linfo_string29                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	9                               # DW_AT_decl_line
	.long	392                             # DW_AT_type
	.byte	7                               # Abbrev [7] 0x16a:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	64
	.long	.Linfo_string30                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	11                              # DW_AT_decl_line
	.long	429                             # DW_AT_type
	.byte	7                               # Abbrev [7] 0x178:0xf DW_TAG_variable
	.byte	3                               # DW_AT_location
	.byte	145
	.ascii	"\274\177"
	.long	.Linfo_string32                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.long	404                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	8                               # Abbrev [8] 0x188:0x5 DW_TAG_pointer_type
	.long	397                             # DW_AT_type
	.byte	4                               # Abbrev [4] 0x18d:0x7 DW_TAG_base_type
	.long	.Linfo_string8                  # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	4                               # Abbrev [4] 0x194:0x7 DW_TAG_base_type
	.long	.Linfo_string11                 # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	4                               # Abbrev [4] 0x19b:0x7 DW_TAG_base_type
	.long	.Linfo_string16                 # DW_AT_name
	.byte	4                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	9                               # Abbrev [9] 0x1a2:0xb DW_TAG_typedef
	.long	42                              # DW_AT_type
	.long	.Linfo_string20                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.byte	10                              # Abbrev [10] 0x1ad:0xc DW_TAG_array_type
	.long	397                             # DW_AT_type
	.byte	11                              # Abbrev [11] 0x1b2:0x6 DW_TAG_subrange_type
	.long	441                             # DW_AT_type
	.byte	16                              # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x1b9:0x7 DW_TAG_base_type
	.long	.Linfo_string31                 # DW_AT_name
	.byte	8                               # DW_AT_byte_size
	.byte	7                               # DW_AT_encoding
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 12.0.0 (/src/external/llvm-project/clang 91c055b1c52a830317cae78986d0d74e433864c6)" # string offset=0
.Linfo_string1:
	.asciz	"print_results.c"               # string offset=97
.Linfo_string2:
	.asciz	"/home/dteixeir/NASA/FT/common" # string offset=113
.Linfo_string3:
	.asciz	"unsigned int"                  # string offset=143
.Linfo_string4:
	.asciz	"false"                         # string offset=156
.Linfo_string5:
	.asciz	"true"                          # string offset=162
.Linfo_string6:
	.asciz	"print_results"                 # string offset=167
.Linfo_string7:
	.asciz	"name"                          # string offset=181
.Linfo_string8:
	.asciz	"char"                          # string offset=186
.Linfo_string9:
	.asciz	"class"                         # string offset=191
.Linfo_string10:
	.asciz	"n1"                            # string offset=197
.Linfo_string11:
	.asciz	"int"                           # string offset=200
.Linfo_string12:
	.asciz	"n2"                            # string offset=204
.Linfo_string13:
	.asciz	"n3"                            # string offset=207
.Linfo_string14:
	.asciz	"niter"                         # string offset=210
.Linfo_string15:
	.asciz	"t"                             # string offset=216
.Linfo_string16:
	.asciz	"double"                        # string offset=218
.Linfo_string17:
	.asciz	"mops"                          # string offset=225
.Linfo_string18:
	.asciz	"optype"                        # string offset=230
.Linfo_string19:
	.asciz	"verified"                      # string offset=237
.Linfo_string20:
	.asciz	"logical"                       # string offset=246
.Linfo_string21:
	.asciz	"npbversion"                    # string offset=254
.Linfo_string22:
	.asciz	"compiletime"                   # string offset=265
.Linfo_string23:
	.asciz	"cs1"                           # string offset=277
.Linfo_string24:
	.asciz	"cs2"                           # string offset=281
.Linfo_string25:
	.asciz	"cs3"                           # string offset=285
.Linfo_string26:
	.asciz	"cs4"                           # string offset=289
.Linfo_string27:
	.asciz	"cs5"                           # string offset=293
.Linfo_string28:
	.asciz	"cs6"                           # string offset=297
.Linfo_string29:
	.asciz	"cs7"                           # string offset=301
.Linfo_string30:
	.asciz	"size"                          # string offset=305
.Linfo_string31:
	.asciz	"__ARRAY_SIZE_TYPE__"           # string offset=310
.Linfo_string32:
	.asciz	"j"                             # string offset=330
	.ident	"clang version 12.0.0 (/src/external/llvm-project/clang 91c055b1c52a830317cae78986d0d74e433864c6)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym printf
	.addrsig_sym sprintf
	.addrsig_sym pow
	.section	.debug_line,"",@progbits
.Lline_table_start0:
