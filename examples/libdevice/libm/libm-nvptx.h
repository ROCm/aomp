//===--------- libm/libm-nvptx.h ------------------------------------------===//
//
//                The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef __LIBM_NVTPX_H__
#define __LIBM_NVTPX_H__

unsigned __nv_brev(unsigned x);
unsigned long long __nv_brevll(unsigned long long x);
int __nv_clz(int x);
int __nv_clzll(long x);
float __nv_fast_cosf(float x);
double __nv_dadd_rd(double __a, double __b);
double __nv_dadd_rn(double __a, double __b);
double __nv_dadd_ru(double __a, double __b);
double __nv_dadd_rz(double __a, double __b);
unsigned int __nv_byte_perm(unsigned int __a, unsigned int __b,
                                       unsigned int __c);
double __nv_ddiv_rd(double __a, double __b);
double __nv_ddiv_rn(double __a, double __b);
double __nv_ddiv_ru(double __a, double __b);
double __nv_ddiv_rz(double __a, double __b);
double __nv_dmul_rd(double __a, double __b);
double __nv_dmul_rn(double __a, double __b);
double __nv_dmul_ru(double __a, double __b);
double __nv_dmul_rz(double __a, double __b);
float __nv_double2float_rd(double __a);
float __nv_double2float_rn(double __a);
float __nv_double2float_ru(double __a);
float __nv_double2float_rz(double __a);
int __nv_double2hiint(double __a);
int __nv_double2int_rd(double __a);
int __nv_double2int_rn(double __a);
int __nv_double2int_ru(double __a);
int __nv_double2int_rz(double __a);
long long __nv_double2ll_rd(double __a);
long long __nv_double2ll_rn(double __a);
long long __nv_double2ll_ru(double __a);
long long __nv_double2ll_rz(double __a);
int __nv_double2loint(double __a);
unsigned int __nv_double2uint_rd(double __a);
unsigned int __nv_double2uint_rn(double __a);
unsigned int __nv_double2uint_ru(double __a);
unsigned int __nv_double2uint_rz(double __a);
unsigned long long __nv_double2ull_rd(double __a);
unsigned long long __nv_double2ull_rn(double __a);
unsigned long long __nv_double2ull_ru(double __a);
unsigned long long __nv_double2ull_rz(double __a);
long long __nv_double_as_longlong(double __a);
double __nv_drcp_rd(double __a);
double __nv_drcp_rn(double __a);
double __nv_drcp_ru(double __a);
double __nv_drcp_rz(double __a);
double __nv_dsqrt_rd(double __a);
double __nv_dsqrt_rn(double __a);
double __nv_dsqrt_ru(double __a);
double __nv_dsqrt_rz(double __a);
double __nv_dsub_rd(double __a, double __b);
double __nv_dsub_rn(double __a, double __b);
double __nv_dsub_ru(double __a, double __b);
double __nv_dsub_rz(double __a, double __b);
float __nv_fast_exp10f(float __a);
float __nv_fast_expf(float __a);
float __nvvm_atom_add_gen_f(volatile float *__p, float __v);
float __nvvm_atom_cta_add_gen_f(volatile float *__p, float __v);
float __nv_fadd_rd(float __a, float __b);
float __nv_fadd_rn(float __a, float __b);
float __nv_fadd_ru(float __a, float __b);
float __nv_fadd_rz(float __a, float __b);
float __nv_fdiv_rd(float __a, float __b);
float __nv_fdiv_rn(float __a, float __b);
float __nv_fdiv_ru(float __a, float __b);
float __nv_fdiv_rz(float __a, float __b);
float __nv_fast_fdividef(float __a, float __b);
int __nv_ffs(int __a);
int __nv_ffsll(long long __a);
int __nv_isfinited(double __a);
int __nv_finitef(float __a);
int __nv_float2int_rd(float __a);
int __nv_float2int_rn(float __a);
int __nv_float2int_ru(float __a);
int __nv_float2int_rz(float __a);
long long __nv_float2ll_rd(float __a);
long long __nv_float2ll_rn(float __a);
long long __nv_float2ll_ru(float __a);
long long __nv_float2ll_rz(float __a);
unsigned int __nv_float2uint_rd(float __a);
unsigned int __nv_float2uint_rn(float __a);
unsigned int __nv_float2uint_ru(float __a);
unsigned int __nv_float2uint_rz(float __a);
unsigned long long __nv_float2ull_rd(float __a);
unsigned long long __nv_float2ull_rn(float __a);
unsigned long long __nv_float2ull_ru(float __a);
unsigned long long __nv_float2ull_rz(float __a);
int __nv_float_as_int(float __a);
unsigned int __nv_float_as_uint(float __a);
double __nv_fma_rd(double __a, double __b, double __c);
double __nv_fma_rn(double __a, double __b, double __c);
double __nv_fma_ru(double __a, double __b, double __c);
double __nv_fma_rz(double __a, double __b, double __c);
float __nv_fmaf_ieee_rd(float __a, float __b, float __c);
float __nv_fmaf_ieee_rn(float __a, float __b, float __c);
float __nv_fmaf_ieee_ru(float __a, float __b, float __c);
float __nv_fmaf_ieee_rz(float __a, float __b, float __c);
float __nv_fmaf_rd(float __a, float __b, float __c);
float __nv_fmaf_rn(float __a, float __b, float __c);
float __nv_fmaf_ru(float __a, float __b, float __c);
float __nv_fmaf_rz(float __a, float __b, float __c);
float __nv_fmul_rd(float __a, float __b);
float __nv_fmul_rn(float __a, float __b);
float __nv_fmul_ru(float __a, float __b);
float __nv_fmul_rz(float __a, float __b);
float __nv_frcp_rd(float __a);
float __nv_frcp_rn(float __a);
float __nv_frcp_ru(float __a);
float __nv_frcp_rz(float __a);
float __nv_frsqrt_rn(float __a);
float __nv_fsqrt_rd(float __a);
float __nv_fsqrt_rn(float __a);
float __nv_fsqrt_ru(float __a);
float __nv_fsqrt_rz(float __a);
float __nv_fsub_rd(float __a, float __b);
float __nv_fsub_rn(float __a, float __b);
float __nv_fsub_ru(float __a, float __b);
float __nv_fsub_rz(float __a, float __b);
unsigned short __nv_hadd(unsigned short __a, unsigned short __b);
double __nv_hiloint2double(int __a, int __b);
int __nvvm_atom_add_gen_i(volatile int *__p, int __v);
int __nvvm_atom_cta_add_gen_i(volatile int *__p, int __v);
int __nvvm_atom_sys_add_gen_i(volatile int *__p, int __v);
int __nvvm_atom_and_gen_i(volatile int *__p, int __v);
int __nvvm_atom_cta_and_gen_i(volatile int *__p, int __v);
int __nvvm_atom_sys_and_gen_i(volatile int *__p, int __v);
int __nvvm_atom_cas_gen_i(volatile int *__p, int __cmp, int __v);
int __nvvm_atom_cta_cas_gen_i(volatile int *__p, int __cmp, int __v);
int __nvvm_atom_sys_cas_gen_i(volatile int *__p, int __cmp, int __v);
int __nvvm_atom_xchg_gen_i(volatile int *__p, int __v);
int __nvvm_atom_cta_xchg_gen_i(volatile int *__p, int __v);
int __nvvm_atom_sys_xchg_gen_i(volatile int *__p, int __v);
int __nvvm_atom_max_gen_i(volatile int *__p, int __v);
int __nvvm_atom_cta_max_gen_i(volatile int *__p, int __v);
int __nvvm_atom_sys_max_gen_i(volatile int *__p, int __v);
int __nvvm_atom_min_gen_i(volatile int *__p, int __v);
int __nvvm_atom_cta_min_gen_i(volatile int *__p, int __v);
int __nvvm_atom_sys_min_gen_i(volatile int *__p, int __v);
int __nvvm_atom_or_gen_i(volatile int *__p, int __v);
int __nvvm_atom_cta_or_gen_i(volatile int *__p, int __v);
int __nvvm_atom_sys_or_gen_i(volatile int *__p, int __v);
int __nvvm_atom_xor_gen_i(volatile int *__p, int __v);
int __nvvm_atom_cta_xor_gen_i(volatile int *__p, int __v);
int __nvvm_atom_sys_xor_gen_i(volatile int *__p, int __v);
long long __nvvm_atom_max_gen_ll(volatile long long *__p,
                                            long long __v);
long long __nvvm_atom_cta_max_gen_ll(volatile long long *__p,
                                                long long __v);
long long __nvvm_atom_sys_max_gen_ll(volatile long long *__p,
                                                long long __v);
long long __nvvm_atom_min_gen_ll(volatile long long *__p,
                                            long long __v);
long long __nvvm_atom_cta_min_gen_ll(volatile long long *__p,
                                                long long __v);
long long __nvvm_atom_sys_min_gen_ll(volatile long long *__p,
                                                long long __v);
double __nv_int2double_rn(int __a);
float __nv_int2float_rd(int __a);
float __nv_int2float_rn(int __a);
float __nv_int2float_ru(int __a);
float __nv_int2float_rz(int __a);
float __nv_int_as_float(int __a);
int __nv_isfinited(double __a);
int __nv_isinfd(double __a);
int __nv_isinff(float __a);
int __nv_isnand(double __a);
int __nv_isnanf(float __a);
double __nv_ll2double_rd(long long __a);
double __nv_ll2double_rn(long long __a);
double __nv_ll2double_ru(long long __a);
double __nv_ll2double_rz(long long __a);
float __nv_ll2float_rd(long long __a);
float __nv_ll2float_rn(long long __a);
float __nv_ll2float_ru(long long __a);
float __nv_ll2float_rz(long long __a);
long long __nvvm_atom_and_gen_ll(volatile long long *__p,
                                            long long __v);
long long __nvvm_atom_cta_and_gen_ll(volatile long long *__p,
                                                long long __v);
long long __nvvm_atom_sys_and_gen_ll(volatile long long *__p,
                                                long long __v);
long long __nvvm_atom_or_gen_ll(volatile long long *__p,
                                           long long __v);
long long __nvvm_atom_cta_or_gen_ll(volatile long long *__p,
                                               long long __v);
long long __nvvm_atom_sys_or_gen_ll(volatile long long *__p,
                                               long long __v);
long long __nvvm_atom_xor_gen_ll(volatile long long *__p,
                                            long long __v);
long long __nvvm_atom_cta_xor_gen_ll(volatile long long *__p,
                                                long long __v);
long long __nvvm_atom_sys_xor_gen_ll(volatile long long *__p,
                                                long long __v);
float __nv_fast_log10f(float __a);
float __nv_fast_log2f(float __a);
float __nv_fast_logf(float __a);
double __nv_longlong_as_double(long long __a);
int __nv_mul24(int __a, int __b);
long long __nv_mul64hi(long long __a, long long __b);
int __nv_mulhi(int __a, int __b);
//unsigned int __nvvm_read_ptx_sreg_pm0(void);
//unsigned int __nvvm_read_ptx_sreg_pm1(void);
//unsigned int __nvvm_read_ptx_sreg_pm2(void);
//unsigned int __nvvm_read_ptx_sreg_pm3(void);
int __nv_popc(int __a);
int __nv_popcll(long long __a);
float __nv_fast_powf(float __a, float __b);
int __nv_rhadd(int __a, int __b);
unsigned int __nv_sad(int __a, int __b, unsigned int __c);
float __nv_saturatef(float __a);
int __nv_signbitd(double __a);
int __nv_signbitf(float __a);
void __nv_fast_sincosf(float __a, float *__sptr, float *__cptr);
float __nv_fast_sinf(float __a);
//int __nvvm_bar0_and(int __a);
//int __nvvm_bar0_popc(int __a);
//int __nvvm_bar0_or(int __a);
float __nv_fast_tanf(float __a);
//void __nvvm_membar_gl();
//void __nvvm_membar_cta();
//void __nvvm_membar_sys();
unsigned int __nvvm_atom_dec_gen_ui(volatile unsigned int *__p,
                                               unsigned int __v);
unsigned int __nvvm_atom_cta_dec_gen_ui(volatile unsigned int *__p,
                                                   unsigned int __v);
unsigned int __nvvm_atom_sys_dec_gen_ui(volatile unsigned int *__p,
                                                   unsigned int __v);
unsigned int __nvvm_atom_inc_gen_ui(volatile unsigned int *__p,
                                               unsigned int __v);
unsigned int __nvvm_atom_cta_inc_gen_ui(volatile unsigned int *__p,
                                                   unsigned int __v);
unsigned int __nvvm_atom_sys_inc_gen_ui(volatile unsigned int *__p,
                                                   unsigned int __v);
unsigned int __nvvm_atom_max_gen_ui(volatile unsigned int *__p,
                                               unsigned int __v);
unsigned int __nvvm_atom_cta_max_gen_ui(volatile unsigned int *__p,
                                                   unsigned int __v);
unsigned int __nvvm_atom_sys_max_gen_ui(volatile unsigned int *__p,
                                                   unsigned int __v);
unsigned int __nvvm_atom_min_gen_ui(volatile unsigned int *__p,
                                               unsigned int __v);
unsigned int __nvvm_atom_cta_min_gen_ui(volatile unsigned int *__p,
                                                   unsigned int __v);
unsigned int __nvvm_atom_sys_min_gen_ui(volatile unsigned int *__p,
                                                   unsigned int __v);
unsigned int __nv_uhadd(unsigned int __a, unsigned int __b);
double __nv_uint2double_rn(unsigned int __a);
float __nv_uint2float_rd(unsigned int __a);
float __nv_uint2float_rn(unsigned int __a);
float __nv_uint2float_ru(unsigned int __a);
float __nv_uint2float_rz(unsigned int __a);
float __nv_uint_as_float(unsigned int __a);
double __nv_ull2double_rd(unsigned long long __a);
double __nv_ull2double_rn(unsigned long long __a);
double __nv_ull2double_ru(unsigned long long __a);
double __nv_ull2double_rz(unsigned long long __a);
float __nv_ull2float_rd(unsigned long long __a);
float __nv_ull2float_rn(unsigned long long __a);
float __nv_ull2float_ru(unsigned long long __a);
float __nv_ull2float_rz(unsigned long long __a);
long long int __nvvm_atom_add_gen_ll(volatile long long int *__p,
                                                long long int __v);
long long int __nvvm_atom_cta_add_gen_ll(volatile long long int *__p,
                                                    long long int __v);
long long int __nvvm_atom_sys_add_gen_ll(volatile long long int *__p,
                                                    long long int __v);
long long int __nvvm_atom_cas_gen_ll(volatile long long int *__p,
                                                long long int __cmp,
                                                long long int __v);
long long int __nvvm_atom_cta_cas_gen_ll(volatile long long int *__p,
                                                    long long int __cmp,
                                                    long long int __v);
long long int __nvvm_atom_sys_cas_gen_ll(volatile long long int *__p,
                                                    long long int __cmp,
                                                    long long int __v);
long long int __nvvm_atom_xchg_gen_ll(volatile long long int *__p,
                                                 long long int __v);
long long int
__nvvm_atom_cta_xchg_gen_ll(volatile long long int *__p, long long int __v);
long long int
__nvvm_atom_sys_xchg_gen_ll(volatile long long int *__p, long long int __v);
unsigned long long
__nvvm_atom_max_gen_ull(volatile unsigned long long *__p,
                        unsigned long long __v);
unsigned long long
__nvvm_atom_cta_max_gen_ull(volatile unsigned long long *__p,
                            unsigned long long __v);
unsigned long long
__nvvm_atom_sys_max_gen_ull(volatile unsigned long long *__p,
                            unsigned long long __v);
unsigned long long
__nvvm_atom_min_gen_ull(volatile unsigned long long *__p,
                        unsigned long long __v);
unsigned long long
__nvvm_atom_cta_min_gen_ull(volatile unsigned long long *__p,
                            unsigned long long __v);
unsigned long long
__nvvm_atom_sys_min_gen_ull(volatile unsigned long long *__p,
                            unsigned long long __v);
unsigned int __nv_umul24(unsigned int __a, unsigned int __b);
unsigned long long __nv_umul64hi(unsigned long long __a,
                                            unsigned long long __b);
unsigned int __nv_umulhi(unsigned int __a, unsigned int __b);
unsigned int __nv_urhadd(unsigned int __a, unsigned int __b);
unsigned int __nv_usad(unsigned int __a, unsigned int __b,
                                  unsigned int __c);
unsigned int __nv_vabs2(unsigned int __a);
unsigned int __nv_vabs4(unsigned int __a);
unsigned int __nv_vabsdiffs2(unsigned int __a, unsigned int __b);
unsigned int __nv_vabsdiffs4(unsigned int __a, unsigned int __b);
unsigned int __nv_vabsdiffu2(unsigned int __a, unsigned int __b);
unsigned int __nv_vabsdiffu4(unsigned int __a, unsigned int __b);
unsigned int __nv_vabsss2(unsigned int __a);
unsigned int __nv_vabsss4(unsigned int __a);
unsigned int __nv_vadd2(unsigned int __a, unsigned int __b);
unsigned int __nv_vadd4(unsigned int __a, unsigned int __b);
unsigned int __nv_vaddss2(unsigned int __a, unsigned int __b);
unsigned int __nv_vaddss4(unsigned int __a, unsigned int __b);
unsigned int __nv_vaddus2(unsigned int __a, unsigned int __b);
unsigned int __nv_vaddus4(unsigned int __a, unsigned int __b);
unsigned int __nv_vavgs2(unsigned int __a, unsigned int __b);
unsigned int __nv_vavgs4(unsigned int __a, unsigned int __b);
unsigned int __nv_vavgu2(unsigned int __a, unsigned int __b);
unsigned int __nv_vavgu4(unsigned int __a, unsigned int __b);
unsigned int __nv_vcmpeq2(unsigned int __a, unsigned int __b);
unsigned int __nv_vcmpeq4(unsigned int __a, unsigned int __b);
unsigned int __nv_vcmpges2(unsigned int __a, unsigned int __b);
unsigned int __nv_vcmpges4(unsigned int __a, unsigned int __b);
unsigned int __nv_vcmpgeu2(unsigned int __a, unsigned int __b);
unsigned int __nv_vcmpgeu4(unsigned int __a, unsigned int __b);
unsigned int __nv_vcmpgts2(unsigned int __a, unsigned int __b);
unsigned int __nv_vcmpgts4(unsigned int __a, unsigned int __b);
unsigned int __nv_vcmpgtu2(unsigned int __a, unsigned int __b);
unsigned int __nv_vcmpgtu4(unsigned int __a, unsigned int __b);
unsigned int __nv_vcmples2(unsigned int __a, unsigned int __b);
unsigned int __nv_vcmples4(unsigned int __a, unsigned int __b);
unsigned int __nv_vcmpleu2(unsigned int __a, unsigned int __b);
unsigned int __nv_vcmpleu4(unsigned int __a, unsigned int __b);
unsigned int __nv_vcmplts2(unsigned int __a, unsigned int __b);
unsigned int __nv_vcmplts4(unsigned int __a, unsigned int __b);
unsigned int __nv_vcmpltu2(unsigned int __a, unsigned int __b);
unsigned int __nv_vcmpltu4(unsigned int __a, unsigned int __b);
unsigned int __nv_vcmpne2(unsigned int __a, unsigned int __b);
unsigned int __nv_vcmpne4(unsigned int __a, unsigned int __b);
unsigned int __nv_vhaddu2(unsigned int __a, unsigned int __b);
unsigned int __nv_vhaddu4(unsigned int __a, unsigned int __b);
unsigned int __nv_vmaxs2(unsigned int __a, unsigned int __b);
unsigned int __nv_vmaxs4(unsigned int __a, unsigned int __b);
unsigned int __nv_vmaxu2(unsigned int __a, unsigned int __b);
unsigned int __nv_vmaxu4(unsigned int __a, unsigned int __b);
unsigned int __nv_vmins2(unsigned int __a, unsigned int __b);
unsigned int __nv_vmins4(unsigned int __a, unsigned int __b);
unsigned int __nv_vminu2(unsigned int __a, unsigned int __b);
unsigned int __nv_vminu4(unsigned int __a, unsigned int __b);
unsigned int __nv_vneg2(unsigned int __a);
unsigned int __nv_vneg4(unsigned int __a);
unsigned int __nv_vnegss2(unsigned int __a);
unsigned int __nv_vnegss4(unsigned int __a);
unsigned int __nv_vsads2(unsigned int __a, unsigned int __b);
unsigned int __nv_vsads4(unsigned int __a, unsigned int __b);
unsigned int __nv_vsadu2(unsigned int __a, unsigned int __b);
unsigned int __nv_vsadu4(unsigned int __a, unsigned int __b);
unsigned int __nv_vseteq2(unsigned int __a, unsigned int __b);
unsigned int __nv_vseteq4(unsigned int __a, unsigned int __b);
unsigned int __nv_vsetges2(unsigned int __a, unsigned int __b);
unsigned int __nv_vsetges4(unsigned int __a, unsigned int __b);
unsigned int __nv_vsetgeu2(unsigned int __a, unsigned int __b);
unsigned int __nv_vsetgeu4(unsigned int __a, unsigned int __b);
unsigned int __nv_vsetgts2(unsigned int __a, unsigned int __b);
unsigned int __nv_vsetgts4(unsigned int __a, unsigned int __b);
unsigned int __nv_vsetgtu2(unsigned int __a, unsigned int __b);
unsigned int __nv_vsetgtu4(unsigned int __a, unsigned int __b);
unsigned int __nv_vsetles2(unsigned int __a, unsigned int __b);
unsigned int __nv_vsetles4(unsigned int __a, unsigned int __b);
unsigned int __nv_vsetleu2(unsigned int __a, unsigned int __b);
unsigned int __nv_vsetleu4(unsigned int __a, unsigned int __b);
unsigned int __nv_vsetlts2(unsigned int __a, unsigned int __b);
unsigned int __nv_vsetlts4(unsigned int __a, unsigned int __b);
unsigned int __nv_vsetltu2(unsigned int __a, unsigned int __b);
unsigned int __nv_vsetltu4(unsigned int __a, unsigned int __b);
unsigned int __nv_vsetne2(unsigned int __a, unsigned int __b);
unsigned int __nv_vsetne4(unsigned int __a, unsigned int __b);
unsigned int __nv_vsub2(unsigned int __a, unsigned int __b);
unsigned int __nv_vsub4(unsigned int __a, unsigned int __b);
unsigned int __nv_vsubss2(unsigned int __a, unsigned int __b);
unsigned int __nv_vsubss4(unsigned int __a, unsigned int __b);
unsigned int __nv_vsubus2(unsigned int __a, unsigned int __b);
unsigned int __nv_vsubus4(unsigned int __a, unsigned int __b);
int _nv_abs(int __a);
double __nv_acos(double __a);
float __nv_acosf(float __a);
double __nv_acosh(double __a);
float __nv_acoshf(float __a);
double __nv_asin(double __a);
float __nv_asinf(float __a);
double __nv_asinh(double __a);
float __nv_asinhf(float __a);
double __nv_atan(double __a);
double __nv_atan2(double __a, double __b);
float __nv_atan2f(float __a, float __b);
float __nv_atanf(float __a);
double __nv_atanh(double __a);
float __nv_atanhf(float __a);
double __nv_cbrt(double __a);
float __nv_cbrtf(float __a);
double __nv_ceil(double __a);
float __nv_ceilf(float __a);
//int __nvvm_read_ptx_sreg_clock();
//long long __nvvm_read_ptx_sreg_clock64();
double __nv_copysign(double __a, double __b);
float __nv_copysignf(float __a, float __b);
double __nv_cos(double __);
float __nv_fast_cosf(float __a);
float __nv_cosf(float __a);
double __nv_cosh(double __a);
float __nv_coshf(float __a);
double __nv_cospi(double __a);
float __nv_cospif(float __a);
double __nv_cyl_bessel_i0(double __a);
float __nv_cyl_bessel_i0f(float __a);
double __nv_cyl_bessel_i1(double __a);
float __nv_cyl_bessel_i1f(float __a);
double __nv_erf(double __a);
double __nv_erfc(double __a);
float __nv_erfcf(float __a);
double __nv_erfcinv(double __a);
float __nv_erfcinvf(float __a);
double __nv_erfcx(double __a);
float __nv_erfcxf(float __a);
float __nv_erff(float __a);
double __nv_erfinv(double __a);
float __nv_erfinvf(float __a);
double __nv_exp(double __a);
double __nv_exp10(double __a);
float __nv_exp10f(float __a);
double __nv_exp2(double __a);
float __nv_exp2f(float __a);
float __nv_expf(float __a);
double __nv_expm1(double __a);
float __nv_expm1f(float __a);
double __nv_fabs(double __a);
float __nv_fabsf(float __a);
double __nv_fdim(double __a, double __b);
float __nv_fdimf(float __a, float __b);
float __nv_fast_fdividef(float __a, float __b);
double __nv_floor(double __f);
float __nv_floorf(float __f);
double __nv_fma(double __a, double __b, double __c);
float __nv_fmaf(float __a, float __b, float __c);
double __nv_fmax(double __a, double __b);
float __nv_fmaxf(float __a, float __b);
double __nv_fmin(double __a, double __b);
float __nv_fminf(float __a, float __b);
double __nv_fmod(double __a, double __b);
float __nv_fmodf(float __a, float __b);
double __nv_frexp(double __a, int *__b);
float __nv_frexpf(float __a, int *__b);
double __nv_hypot(double __a, double __b);
float __nv_hypotf(float __a, float __b);
int __nv_ilogb(double __a);
int __nv_ilogbf(float __a);
double __nv_j0(double __a);
float __nv_j0f(float __a);
double __nv_j1(double __a);
float __nv_j1f(float __a);
double __nv_jn(int __n, double __a);
float __nv_jnf(int __n, float __a);
long __nv_abs(long __a);
double __nv_ldexp(double __a, int __b);
float __nv_ldexpf(float __a, int __b);
double __nv_lgamma(double __a);
float __nv_lgammaf(float __a);
long long __nv_llabs(long long __a);
long long __nv_llmax(long long __a, long long __b);
long long __nv_llmin(long long __a, long long __b);
long long __nv_llrint(double __a);
long long __nv_llrintf(float __a);
long long __nv_llround(double __a);
long long __nv_llroundf(float __a);
double __nv_log(double __a);
double __nv_log10(double __a);
float __nv_log10f(float __a);
double __nv_log1p(double __a);
float __nv_log1pf(float __a);
double __nv_log2(double __a);
float __nv_log2f(float __a);
float __nv_fast_log2f(float __a);
double __nv_logb(double __a);
float __nv_logbf(float __a);
float __nv_fast_logf(float __a);
float __nv_logf(float __a);
int __nv_max(int __a, int __b);
int __nv_min(int __a, int __b);
double __nv_modf(double __a, double *__b);
float __nv_modff(float __a, float *__b);
double __nv_nearbyint(double __a);
float __nv_nearbyintf(float __a);
double __nv_nextafter(double __a, double __b);
float __nv_nextafterf(float __a, float __b);
double __nv_norm(int __dim, const double *__t);
double __nv_norm3d(double __a, double __b, double __c);
float __nv_norm3df(float __a, float __b, float __c);
double __nv_norm4d(double __a, double __b, double __c, double __d);
float __nv_norm4df(float __a, float __b, float __c, float __d);
double __nv_normcdf(double __a);
float __nv_normcdff(float __a);
double __nv_normcdfinv(double __a);
float __nv_normcdfinvf(float __a);
float __nv_normf(int __dim, const float *__t);

double __nv_pow(double __a, double __b);
float __nv_powf(float __a, float __b);
double __nv_powi(double __a, int __b);
float __nv_powif(float __a, int __b);
double __nv_rcbrt(double __a);
float __nv_rcbrtf(float __a);
double __nv_remainder(double __a, double __b);
float __nv_remainderf(float __a, float __b);
double __nv_remquo(double __a, double __b, int *__c);
float __nv_remquof(float __a, float __b, int *__c);
double __nv_rhypot(double __a, double __b);
float __nv_rhypotf(float __a, float __b);
double __nv_rint(double __a);
float __nv_rintf(float __a);
double __nv_rnorm(int __a, const double *__b);
double __nv_rnorm3d(double __a, double __b, double __c);
float __nv_rnorm3df(float __a, float __b, float __c);
double __nv_rnorm4d(double __a, double __b, double __c, double __d);
float __nv_rnorm4df(float __a, float __b, float __c, float __d);
float __nv_rnormf(int __dim, const float *__t);
double __nv_round(double __a);
float __nv_roundf(float __a);
double __nv_rsqrt(double __a);
float __nv_rsqrtf(float __a);
double __nv_scalbn(double __a, int __b);
float __nv_scalbnf(float __a, int __b);
double __nv_sin(double __a);
void __nv_sincos(double __a, double *__sptr, double *__cptr);
void __nv_sincosf(float __a, float *__sptr, float *__cptr);
void __nv_sincospi(double __a, double *__sptr, double *__cptr);
void __nv_sincospif(float __a, float *__sptr, float *__cptr);
float __nv_fast_sinf(float __a);
float __nv_sinf(float __a);
double __nv_sinh(double __a);
float __nv_sinhf(float __a);
double __nv_sinpi(double __a);
float __nv_sinpif(float __a);
double __nv_sqrt(double __a);
float __nv_sqrtf(float __a);
double __nv_tan(double __a);
float __nv_tanf(float __a);
double __nv_tanh(double __a);
float __nv_tanhf(float __a);
double __nv_tgamma(double __a);
float __nv_tgammaf(float __a);
double __nv_trunc(double __a);
float __nv_truncf(float __a);
unsigned long long __nv_ullmax(unsigned long long __a,
                                          unsigned long long __b);
unsigned long long __nv_ullmin(unsigned long long __a,
                                          unsigned long long __b);
unsigned int __nv_umax(unsigned int __a, unsigned int __b);
unsigned int __nv_umin(unsigned int __a, unsigned int __b);
double __nv_y0(double __a);
float __nv_y0f(float __a);
double __nv_y1(double __a);
float __nv_y1f(float __a);
double __nv_yn(int __a, double __b);
float __nv_ynf(int __a, float __b);

#endif // __LIBM_NVTPX_H__
