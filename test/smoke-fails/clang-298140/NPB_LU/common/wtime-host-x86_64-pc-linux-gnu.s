	.text
	.file	"wtime.c"
	.file	1 "/home/dteixeir/NASA/FT/common" "../common/wtime.c"
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3                               # -- Begin function wtime_
.LCPI0_0:
	.quad	0x3eb0c6f7a0b5ed8d              # double 9.9999999999999995E-7
	.text
	.globl	wtime_
	.p2align	4, 0x90
	.type	wtime_,@function
wtime_:                                 # @wtime_
.Lfunc_begin0:
	.loc	1 8 0                           # ../common/wtime.c:8:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	xorl	%eax, %eax
	movl	%eax, %esi
	leaq	-24(%rbp), %rdi
.Ltmp0:
	.loc	1 11 3 prologue_end             # ../common/wtime.c:11:3
	callq	gettimeofday
.Ltmp1:
	.loc	1 12 7                          # ../common/wtime.c:12:7
	movabsq	$wtime_.sec, %rax
	movl	(%rax), %eax
	.loc	1 12 11 is_stmt 0               # ../common/wtime.c:12:11
	cmpl	$0, %eax
.Ltmp2:
	.loc	1 12 7                          # ../common/wtime.c:12:7
	jge	.LBB0_2
# %bb.1:
.Ltmp3:
	.loc	1 12 25                         # ../common/wtime.c:12:25
	movl	-24(%rbp), %ecx
	.loc	1 12 20                         # ../common/wtime.c:12:20
	movabsq	$wtime_.sec, %rax
	movl	%ecx, (%rax)
.Ltmp4:
.LBB0_2:
	.loc	1 13 12 is_stmt 1               # ../common/wtime.c:13:12
	movq	-24(%rbp), %rax
	.loc	1 13 21 is_stmt 0               # ../common/wtime.c:13:21
	movabsq	$wtime_.sec, %rcx
	movslq	(%rcx), %rcx
	.loc	1 13 19                         # ../common/wtime.c:13:19
	subq	%rcx, %rax
	.loc	1 13 8                          # ../common/wtime.c:13:8
	cvtsi2sd	%rax, %xmm0
	.loc	1 13 35                         # ../common/wtime.c:13:35
	cvtsi2sdq	-16(%rbp), %xmm1
	movabsq	$.LCPI0_0, %rax
	movsd	(%rax), %xmm2                   # xmm2 = mem[0],zero
	.loc	1 13 34                         # ../common/wtime.c:13:34
	mulsd	%xmm2, %xmm1
	.loc	1 13 26                         # ../common/wtime.c:13:26
	addsd	%xmm1, %xmm0
	.loc	1 13 4                          # ../common/wtime.c:13:4
	movq	-8(%rbp), %rax
	.loc	1 13 6                          # ../common/wtime.c:13:6
	movsd	%xmm0, (%rax)
	.loc	1 14 1 is_stmt 1                # ../common/wtime.c:14:1
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp5:
.Lfunc_end0:
	.size	wtime_, .Lfunc_end0-wtime_
	.cfi_endproc
                                        # -- End function
	.type	wtime_.sec,@object              # @wtime_.sec
	.data
	.p2align	2
wtime_.sec:
	.long	4294967295                      # 0xffffffff
	.size	wtime_.sec, 4

	.file	2 "/usr/include/bits" "types.h"
	.file	3 "/usr/include/bits/types" "struct_timeval.h"
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
	.byte	3                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
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
	.byte	5                               # Abbreviation Code
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
	.byte	6                               # Abbreviation Code
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
	.byte	7                               # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	8                               # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	9                               # Abbreviation Code
	.byte	13                              # DW_TAG_member
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	56                              # DW_AT_data_member_location
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	10                              # Abbreviation Code
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
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0xb8 DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	12                              # DW_AT_language
	.long	.Linfo_string1                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string2                  # DW_AT_comp_dir
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	2                               # Abbrev [2] 0x2a:0x47 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string5                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
                                        # DW_AT_prototyped
                                        # DW_AT_external
	.byte	3                               # Abbrev [3] 0x3f:0x15 DW_TAG_variable
	.long	.Linfo_string3                  # DW_AT_name
	.long	113                             # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	9                               # DW_AT_decl_line
	.byte	9                               # DW_AT_location
	.byte	3
	.quad	wtime_.sec
	.byte	4                               # Abbrev [4] 0x54:0xe DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string6                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.long	120                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x62:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	104
	.long	.Linfo_string8                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	10                              # DW_AT_decl_line
	.long	132                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	6                               # Abbrev [6] 0x71:0x7 DW_TAG_base_type
	.long	.Linfo_string4                  # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	7                               # Abbrev [7] 0x78:0x5 DW_TAG_pointer_type
	.long	125                             # DW_AT_type
	.byte	6                               # Abbrev [6] 0x7d:0x7 DW_TAG_base_type
	.long	.Linfo_string7                  # DW_AT_name
	.byte	4                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	8                               # Abbrev [8] 0x84:0x21 DW_TAG_structure_type
	.long	.Linfo_string14                 # DW_AT_name
	.byte	16                              # DW_AT_byte_size
	.byte	3                               # DW_AT_decl_file
	.byte	8                               # DW_AT_decl_line
	.byte	9                               # Abbrev [9] 0x8c:0xc DW_TAG_member
	.long	.Linfo_string9                  # DW_AT_name
	.long	165                             # DW_AT_type
	.byte	3                               # DW_AT_decl_file
	.byte	10                              # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	9                               # Abbrev [9] 0x98:0xc DW_TAG_member
	.long	.Linfo_string12                 # DW_AT_name
	.long	183                             # DW_AT_type
	.byte	3                               # DW_AT_decl_file
	.byte	11                              # DW_AT_decl_line
	.byte	8                               # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0xa5:0xb DW_TAG_typedef
	.long	176                             # DW_AT_type
	.long	.Linfo_string11                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	158                             # DW_AT_decl_line
	.byte	6                               # Abbrev [6] 0xb0:0x7 DW_TAG_base_type
	.long	.Linfo_string10                 # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	10                              # Abbrev [10] 0xb7:0xb DW_TAG_typedef
	.long	176                             # DW_AT_type
	.long	.Linfo_string13                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	160                             # DW_AT_decl_line
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 12.0.0 (/src/external/llvm-project/clang 91c055b1c52a830317cae78986d0d74e433864c6)" # string offset=0
.Linfo_string1:
	.asciz	"wtime.c"                       # string offset=97
.Linfo_string2:
	.asciz	"/home/dteixeir/NASA/FT/common" # string offset=105
.Linfo_string3:
	.asciz	"sec"                           # string offset=135
.Linfo_string4:
	.asciz	"int"                           # string offset=139
.Linfo_string5:
	.asciz	"wtime_"                        # string offset=143
.Linfo_string6:
	.asciz	"t"                             # string offset=150
.Linfo_string7:
	.asciz	"double"                        # string offset=152
.Linfo_string8:
	.asciz	"tv"                            # string offset=159
.Linfo_string9:
	.asciz	"tv_sec"                        # string offset=162
.Linfo_string10:
	.asciz	"long int"                      # string offset=169
.Linfo_string11:
	.asciz	"__time_t"                      # string offset=178
.Linfo_string12:
	.asciz	"tv_usec"                       # string offset=187
.Linfo_string13:
	.asciz	"__suseconds_t"                 # string offset=195
.Linfo_string14:
	.asciz	"timeval"                       # string offset=209
	.ident	"clang version 12.0.0 (/src/external/llvm-project/clang 91c055b1c52a830317cae78986d0d74e433864c6)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym gettimeofday
	.addrsig_sym wtime_.sec
	.section	.debug_line,"",@progbits
.Lline_table_start0:
