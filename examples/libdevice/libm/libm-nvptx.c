//===--------- libm/libm-nvptx.c ------------------------------------------===//
//
//                The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <math.h>
#include <stddef.h>
#include <limits.h>
#include "libm-nvptx.h"
#pragma omp declare target

#if 0
#define __FAST_OR_SLOW(fast, slow) fast
#else
#define __FAST_OR_SLOW(fast, slow) slow
#endif

// BEGIN FLOAT
float acosf(float __a) { return __nv_acosf(__a); }

float acoshf(float __a) { return __nv_acoshf(__a); }

float asinf(float __a) { return __nv_asinf(__a); }

float asinhf(float __a) { return __nv_asinhf(__a); }

float atan2f(float __a, float __b) { return __nv_atan2f(__a, __b); }

float atanf(float __a) { return __nv_atanf(__a); }

float atanhf(float __a) { return __nv_atanhf(__a); }

float cbrtf(float __a) { return __nv_cbrtf(__a); }

float ceilf(float __a) { return __nv_ceilf(__a); }

float copysignf(float __a, float __b) {
  return __nv_copysignf(__a, __b);
}

float cosf(float __a) {
  return __FAST_OR_SLOW(__nv_fast_cosf, __nv_cosf)(__a);
}

float coshf(float __a) { return __nv_coshf(__a); }

float cospif(float __a) { return __nv_cospif(__a); }

float cyl_bessel_i0f(float __a) { return __nv_cyl_bessel_i0f(__a); }

float cyl_bessel_i1f(float __a) { return __nv_cyl_bessel_i1f(__a); }

float erfcf(float __a) { return __nv_erfcf(__a); }

float erfcinvf(float __a) { return __nv_erfcinvf(__a); }

float erfcxf(float __a) { return __nv_erfcxf(__a); }

float erff(float __a) { return __nv_erff(__a); }

float erfinvf(float __a) { return __nv_erfinvf(__a); }

float exp10f(float __a) { return __nv_exp10f(__a); }

float exp2f(float __a) { return __nv_exp2f(__a); }

float expf(float __a) { return __nv_expf(__a); }

float expm1f(float __a) { return __nv_expm1f(__a); }

float fabsf(float __a) { return __nv_fabsf(__a); }

float fdimf(float __a, float __b) { return __nv_fdimf(__a, __b); }

float fdividef(float __a, float __b) {
#if __FAST_MATH__ && !__CUDA_PREC_DIV
  return __nv_fast_fdividef(__a, __b);
#else
  return __a / __b;
#endif
}

float floorf(float __f) { return __nv_floorf(__f); }

float fmaf(float __a, float __b, float __c) {
  return __nv_fmaf(__a, __b, __c);
}

float fmaxf(float __a, float __b) { return __nv_fmaxf(__a, __b); }

float fminf(float __a, float __b) { return __nv_fminf(__a, __b); }

float fmodf(float __a, float __b) { return __nv_fmodf(__a, __b); }

float frexpf(float __a, int *__b) { return __nv_frexpf(__a, __b); }

float hypotf(float __a, float __b) { return __nv_hypotf(__a, __b); }

int ilogbf(float __a) { return __nv_ilogbf(__a); }

int __finitef(float __a) { return __nv_finitef(__a); }

int __isinff(float __a) { return __nv_isinff(__a); }

int __isnanf(float __a) { return __nv_isnanf(__a); }

float j0f(float __a) { return __nv_j0f(__a); }

float j1f(float __a) { return __nv_j1f(__a); }

float jnf(int __n, float __a) { return __nv_jnf(__n, __a); }

float ldexpf(float __a, int __b) { return __nv_ldexpf(__a, __b); }

float lgammaf(float __a) { return __nv_lgammaf(__a); }

long long llrintf(float __a) { return __nv_llrintf(__a); }

long long llroundf(float __a) { return __nv_llroundf(__a); }

float log10f(float __a) { return __nv_log10f(__a); }

float log1pf(float __a) { return __nv_log1pf(__a); }

float log2f(float __a) {
  return __FAST_OR_SLOW(__nv_fast_log2f, __nv_log2f)(__a);
}

float logbf(float __a) { return __nv_logbf(__a); }

float logf(float __a) {
  return __FAST_OR_SLOW(__nv_fast_logf, __nv_logf)(__a);
}

long long __float2ll_rn(float __a) { return __nv_float2ll_rn(__a); }

#if defined(__LP64__)
long lrintf(float __a) { return __float2ll_rn(__a); }
long lroundf(float __a) { return llroundf(__a); }
#else
long lrintf(float __a) { return __float2int_rn(__a); }
long lroundf(float __a) { return roundf(__a); }
#endif

float modff(float __a, float *__b) { return __nv_modff(__a, __b); }

// nanf - missing

float nearbyintf(float __a) { return __nv_nearbyintf(__a); }

float nextafterf(float __a, float __b) {
  return __nv_nextafterf(__a, __b);
}

float norm3df(float __a, float __b, float __c) {
  return __nv_norm3df(__a, __b, __c);
}

float norm4df(float __a, float __b, float __c, float __d) {
  return __nv_norm4df(__a, __b, __c, __d);
}

float normcdff(float __a) { return __nv_normcdff(__a); }

float normf(int __dim, const float *__t) {
  return __nv_normf(__dim, __t);
}

float powf(float __a, float __b) { return __nv_powf(__a, __b); }

float rcbrtf(float __a) { return __nv_rcbrtf(__a); }

float remainderf(float __a, float __b) {
  return __nv_remainderf(__a, __b);
}

float remquof(float __a, float __b, int *__c) {
  return __nv_remquof(__a, __b, __c);
}

float rhypotf(float __a, float __b) {
  return __nv_rhypotf(__a, __b);
}

float rintf(float __a) { return __nv_rintf(__a); }

float rnorm3df(float __a, float __b, float __c) {
  return __nv_rnorm3df(__a, __b, __c);
}

float rnorm4df(float __a, float __b, float __c, float __d) {
  return __nv_rnorm4df(__a, __b, __c, __d);
}

float normcdfinvf(float __a) { return __nv_normcdfinvf(__a); }

float rnormf(int __dim, const float *__t) {
  return __nv_rnormf(__dim, __t);
}

float roundf(float __a) { return __nv_roundf(__a); }

float rsqrtf(float __a) { return __nv_rsqrtf(__a); }

float scalblnf(float __a, long __b) {
  if (__b > INT_MAX)
    return __a > 0 ? HUGE_VALF : -HUGE_VALF;
  if (__b < INT_MIN)
    return __a > 0 ? 0.f : -0.f;
  return scalbnf(__a, (int)__b);
}

float scalbnf(float __a, int __b) { return __nv_scalbnf(__a, __b); }

int __signbitf(float __a) { return __nv_signbitf(__a); }

void sincosf(float __a, float *__sptr, float *__cptr) {
  return __FAST_OR_SLOW(__nv_fast_sincosf, __nv_sincosf)(__a, __sptr, __cptr);
}

void sincospif(float __a, float *__sptr, float *__cptr) {
  return __nv_sincospif(__a, __sptr, __cptr);
}

float sinf(float __a) {
  return __FAST_OR_SLOW(__nv_fast_sinf, __nv_sinf)(__a);
}

float sinhf(float __a) { return __nv_sinhf(__a); }

float sinpif(float __a) { return __nv_sinpif(__a); }

float sqrtf(float __a) { return __nv_sqrtf(__a); }

float tanf(float __a) { return __nv_tanf(__a); }

float tanhf(float __a) { return __nv_tanhf(__a); }

float tgammaf(float __a) { return __nv_tgammaf(__a); }

float truncf(float __a) { return __nv_truncf(__a); }

float y0f(float __a) { return __nv_y0f(__a); }

float y1f(float __a) { return __nv_y1f(__a); }

float ynf(int __a, float __b) { return __nv_ynf(__a, __b); }

// BEGIN INTRINSICS

float __cosf(float __a) { return __nv_fast_cosf(__a); }

float __exp10f(float __a) { return __nv_fast_exp10f(__a); }

float __expf(float __a) { return __nv_fast_expf(__a); }

float __fadd_rd(float __a, float __b) {
  return __nv_fadd_rd(__a, __b);
}
float __fadd_rn(float __a, float __b) {
  return __nv_fadd_rn(__a, __b);
}
float __fadd_ru(float __a, float __b) {
  return __nv_fadd_ru(__a, __b);
}
float __fadd_rz(float __a, float __b) {
  return __nv_fadd_rz(__a, __b);
}
float __fdiv_rd(float __a, float __b) {
  return __nv_fdiv_rd(__a, __b);
}
float __fdiv_rn(float __a, float __b) {
  return __nv_fdiv_rn(__a, __b);
}
float __fdiv_ru(float __a, float __b) {
  return __nv_fdiv_ru(__a, __b);
}
float __fdiv_rz(float __a, float __b) {
  return __nv_fdiv_rz(__a, __b);
}

float __fdividef(float __a, float __b) {
  return __nv_fast_fdividef(__a, __b);
}

float __fmaf_rd(float __a, float __b, float __c) {
  return __nv_fmaf_rd(__a, __b, __c);
}

float __fmaf_rn(float __a, float __b, float __c) {
  return __nv_fmaf_rn(__a, __b, __c);
}

float __fmaf_ru(float __a, float __b, float __c) {
  return __nv_fmaf_ru(__a, __b, __c);
}

float __fmaf_rz(float __a, float __b, float __c) {
  return __nv_fmaf_rz(__a, __b, __c);
}

float __fmul_rd(float __a, float __b) {
  return __nv_fmul_rd(__a, __b);
}
float __fmul_rn(float __a, float __b) {
  return __nv_fmul_rn(__a, __b);
}
float __fmul_ru(float __a, float __b) {
  return __nv_fmul_ru(__a, __b);
}
float __fmul_rz(float __a, float __b) {
  return __nv_fmul_rz(__a, __b);
}

float __frcp_rd(float __a) { return __nv_frcp_rd(__a); }

float __frcp_rn(float __a) { return __nv_frcp_rn(__a); }

float __frcp_ru(float __a) { return __nv_frcp_ru(__a); }

float __frcp_rz(float __a) { return __nv_frcp_rz(__a); }

float __fsqrt_rd(float __a) { return __nv_fsqrt_rd(__a); }

float __fsqrt_rn(float __a) { return __nv_fsqrt_rn(__a); }

float __fsqrt_ru(float __a) { return __nv_fsqrt_ru(__a); }

float __fsqrt_rz(float __a) { return __nv_fsqrt_rz(__a); }

float __fsub_rd(float __a, float __b) {
  return __nv_fsub_rd(__a, __b);
}

float __fsub_rn(float __a, float __b) {
  return __nv_fsub_rn(__a, __b);
}

float __fsub_ru(float __a, float __b) {
  return __nv_fsub_ru(__a, __b);
}

float __fsub_rz(float __a, float __b) {
  return __nv_fsub_rz(__a, __b);
}

float __log10f(float __a) { return __nv_fast_log10f(__a); }

float __log2f(float __a) { return __nv_fast_log2f(__a); }

float __logf(float __a) { return __nv_fast_logf(__a); }

float __powf(float __a, float __b) {
  return __nv_fast_powf(__a, __b);
}

float __saturatef(float __a) { return __nv_saturatef(__a); }

void __sincosf(float __a, float *__sptr, float *__cptr) {
  return __nv_fast_sincosf(__a, __sptr, __cptr);
}
float __sinf(float __a) { return __nv_fast_sinf(__a); }

float __tanf(float __a) { return __nv_fast_tanf(__a); }

// BEGIN DOUBLE


double acos(double __a) { return __nv_acos(__a); }

double acosh(double __a) { return __nv_acosh(__a); }

double asin(double __a) { return __nv_asin(__a); }

double asinh(double __a) { return __nv_asinh(__a); }

double atan(double __a) { return __nv_atan(__a); }

double atan2(double __a, double __b) { return __nv_atan2(__a, __b); }

double atanh(double __a) { return __nv_atanh(__a); }

double cbrt(double __a) { return __nv_cbrt(__a); }

double ceil(double __a) { return __nv_ceil(__a); }

double copysign(double __a, double __b) {
  return __nv_copysign(__a, __b);
}

double cos(double __a) { return __nv_cos(__a); }

double cosh(double __a) { return __nv_cosh(__a); }

double cospi(double __a) { return __nv_cospi(__a); }

double cyl_bessel_i0(double __a) { return __nv_cyl_bessel_i0(__a); }

double cyl_bessel_i1(double __a) { return __nv_cyl_bessel_i1(__a); }

double erf(double __a) { return __nv_erf(__a); }

double erfc(double __a) { return __nv_erfc(__a); }

double erfcinv(double __a) { return __nv_erfcinv(__a); }

double erfcx(double __a) { return __nv_erfcx(__a); }

double erfinv(double __a) { return __nv_erfinv(__a); }

double exp(double __a) { return __nv_exp(__a); }

double exp10(double __a) { return __nv_exp10(__a); }

double exp2(double __a) { return __nv_exp2(__a); }

double expm1(double __a) { return __nv_expm1(__a); }

double fabs(double __a) { return __nv_fabs(__a); }

double fdim(double __a, double __b) { return __nv_fdim(__a, __b); }

double floor(double __f) { return __nv_floor(__f); }

double fma(double __a, double __b, double __c) {
  return __nv_fma(__a, __b, __c);
}

double fmax(double __a, double __b) { return __nv_fmax(__a, __b); }

double fmin(double __a, double __b) { return __nv_fmin(__a, __b); }

double fmod(double __a, double __b) { return __nv_fmod(__a, __b); }

double frexp(double __a, int *__b) { return __nv_frexp(__a, __b); }

double hypot(double __a, double __b) { return __nv_hypot(__a, __b); }

int ilogb(double __a) { return __nv_ilogb(__a); }

int __finite(double __a) { return __nv_isfinited(__a); }

// These symbols, isnan and isinf, cause issues on some RHEL systems.
//int __isinf(double __a) { return __nv_isinfd(__a); }

//int __isnan(double __a) { return __nv_isnand(__a); }

double j0(double __a) { return __nv_j0(__a); }

double j1(double __a) { return __nv_j1(__a); }

double jn(int __n, double __a) { return __nv_jn(__n, __a); }

double ldexp(double __a, int __b) { return __nv_ldexp(__a, __b); }

double lgamma(double __a) { return __nv_lgamma(__a); }

long long llrint(double __a) { return __nv_llrint(__a); }

long long llround(double __a) { return __nv_llround(__a); }

double log(double __a) { return __nv_log(__a); }

double log10(double __a) { return __nv_log10(__a); }

double log1p(double __a) { return __nv_log1p(__a); }

double log2(double __a) { return __nv_log2(__a); }

double logb(double __a) { return __nv_logb(__a); }

#if defined(__LP64__)
long lrint(double __a) { return llrint(__a); }

long lround(double __a) { return llround(__a); }
#else
long lrint(double __a) { return (long)rint(__a); }

long lround(double __a) { return round(__a); }
#endif

double modf(double __a, double *__b) { return __nv_modf(__a, __b); }

// nan - missing

double nearbyint(double __a) { return __nv_nearbyint(__a); }

double nextafter(double __a, double __b) {
  return __nv_nextafter(__a, __b);
}

double norm(int __dim, const double *__t) {
  return __nv_norm(__dim, __t);
}

double norm3d(double __a, double __b, double __c) {
  return __nv_norm3d(__a, __b, __c);
}

double norm4d(double __a, double __b, double __c, double __d) {
  return __nv_norm4d(__a, __b, __c, __d);
}

double normcdf(double __a) { return __nv_normcdf(__a); }

double normcdfinv(double __a) { return __nv_normcdfinv(__a); }

double pow(double __a, double __b) { return __nv_pow(__a, __b); }

double rcbrt(double __a) { return __nv_rcbrt(__a); }

double remainder(double __a, double __b) {
  return __nv_remainder(__a, __b);
}

double remquo(double __a, double __b, int *__c) {
  return __nv_remquo(__a, __b, __c);
}

double rhypot(double __a, double __b) {
  return __nv_rhypot(__a, __b);
}

double rint(double __a) { return __nv_rint(__a); }

double rnorm(int __a, const double *__b) {
  return __nv_rnorm(__a, __b);
}

double rnorm3d(double __a, double __b, double __c) {
  return __nv_rnorm3d(__a, __b, __c);
}

double rnorm4d(double __a, double __b, double __c, double __d) {
  return __nv_rnorm4d(__a, __b, __c, __d);
}

double round(double __a) { return __nv_round(__a); }

double rsqrt(double __a) { return __nv_rsqrt(__a); }

double scalbn(double __a, int __b) { return __nv_scalbn(__a, __b); }

double scalbln(double __a, long __b) {
  if (__b > INT_MAX)
    return __a > 0 ? HUGE_VAL : -HUGE_VAL;
  if (__b < INT_MIN)
    return __a > 0 ? 0.0 : -0.0;
  return scalbn(__a, (int)__b);
}

int __signbit(double __a) { return __nv_signbitd(__a); }

double sin(double __a) { return __nv_sin(__a); }

void sincos(double __a, double *__sptr, double *__cptr) {
  return __nv_sincos(__a, __sptr, __cptr);
}

void sincospi(double __a, double *__sptr, double *__cptr) {
  return __nv_sincospi(__a, __sptr, __cptr);
}

double sinh(double __a) { return __nv_sinh(__a); }

double sinpi(double __a) { return __nv_sinpi(__a); }

double sqrt(double __a) { return __nv_sqrt(__a); }

double tan(double __a) { return __nv_tan(__a); }

double tanh(double __a) { return __nv_tanh(__a); }

double tgamma(double __a) { return __nv_tgamma(__a); }

double trunc(double __a) { return __nv_trunc(__a); }

double y0(double __a) { return __nv_y0(__a); }

double y1(double __a) { return __nv_y1(__a); }

double yn(int __a, double __b) { return __nv_yn(__a, __b); }

// BEGIN INTRINSICS

double __dadd_rd(double __a, double __b) {
  return __nv_dadd_rd(__a, __b);
}

double __dadd_rn(double __a, double __b) {
  return __nv_dadd_rn(__a, __b);
}

double __dadd_ru(double __a, double __b) {
  return __nv_dadd_ru(__a, __b);
}

double __dadd_rz(double __a, double __b) {
  return __nv_dadd_rz(__a, __b);
}

double __ddiv_rd(double __a, double __b) {
  return __nv_ddiv_rd(__a, __b);
}

double __ddiv_rn(double __a, double __b) {
  return __nv_ddiv_rn(__a, __b);
}

double __ddiv_ru(double __a, double __b) {
  return __nv_ddiv_ru(__a, __b);
}

double __ddiv_rz(double __a, double __b) {
  return __nv_ddiv_rz(__a, __b);
}

double __dmul_rd(double __a, double __b) {
  return __nv_dmul_rd(__a, __b);
}

double __dmul_rn(double __a, double __b) {
  return __nv_dmul_rn(__a, __b);
}

double __dmul_ru(double __a, double __b) {
  return __nv_dmul_ru(__a, __b);
}

double __dmul_rz(double __a, double __b) {
  return __nv_dmul_rz(__a, __b);
}

double __drcp_rd(double __a) { return __nv_drcp_rd(__a); }

double __drcp_rn(double __a) { return __nv_drcp_rn(__a); }

double __drcp_ru(double __a) { return __nv_drcp_ru(__a); }

double __drcp_rz(double __a) { return __nv_drcp_rz(__a); }

double __dsqrt_rd(double __a) { return __nv_dsqrt_rd(__a); }

double __dsqrt_rn(double __a) { return __nv_dsqrt_rn(__a); }

double __dsqrt_ru(double __a) { return __nv_dsqrt_ru(__a); }

double __dsqrt_rz(double __a) { return __nv_dsqrt_rz(__a); }

double __dsub_rd(double __a, double __b) {
  return __nv_dsub_rd(__a, __b);
}

double __dsub_rn(double __a, double __b) {
  return __nv_dsub_rn(__a, __b);
}

double __dsub_ru(double __a, double __b) {
  return __nv_dsub_ru(__a, __b);
}

double __dsub_rz(double __a, double __b) {
  return __nv_dsub_rz(__a, __b);
}

double __fma_rd(double __a, double __b, double __c) {
  return __nv_fma_rd(__a, __b, __c);
}

double __fma_rn(double __a, double __b, double __c) {
  return __nv_fma_rn(__a, __b, __c);
}

double __fma_ru(double __a, double __b, double __c) {
  return __nv_fma_ru(__a, __b, __c);
}

double __fma_rz(double __a, double __b, double __c) {
  return __nv_fma_rz(__a, __b, __c);
}

// END DOUBLE

#pragma omp end declare target
