//===--------- libm/libm-nvptx.cpp ----------------------------------------===//
//
//                The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <cmath>
#include <limits.h>
#include <limits>
#include "libm-nvptx.h"
#ifdef __cplusplus
#pragma omp declare target

// Extern declarations
extern long long llabs(long long __a);
extern long labs(long __a);
extern float powif(float __a, int __b);
extern double powi(double __a, int __b);
extern float powif(float __a, int __b);
extern int __isfinited(double __a);

/// Do not provide the double overload because the base functions
/// are defined in libm-amdgcn.c.  Only provide non base variants.

long long abs(long long __n) { return llabs(__n); }
long abs(long __n) { return labs(__n); }
float abs(float __x) { return fabsf(__x); }
double abs(double __x) { return fabs(__x); }
float acos(float __x) { return acosf(__x); }
float asin(float __x) { return asinf(__x); }
float atan(float __x) { return atanf(__x); }
float atan2(float __x, float __y) { return atan2f(__x, __y); }
float ceil(float __x) { return ceilf(__x); }
float cos(float __x) { return cosf(__x); }
float cosh(float __x) { return coshf(__x); }
float exp(float __x) { return expf(__x); }
float fabs(float __x) { return fabsf(__x); }
float floor(float __x) { return floorf(__x); }
float fmod(float __x, float __y) { return fmodf(__x, __y); }
int fpclassify(float __x) {
  return __builtin_fpclassify(FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL,
                              FP_ZERO, __x);
}
int fpclassify(double __x) {
  return __builtin_fpclassify(FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL,
                              FP_ZERO, __x);
}
float frexp(float __arg, int *__exp) {
  return frexpf(__arg, __exp);
}

bool isinf(float __x) { return __isinff(__x); }
//bool isinf(double __x) { return __isinf(__x); }
bool isfinite(float __x) { return __finitef(__x); }
bool isfinite(double __x) { return __isfinited(__x); }
bool isnan(float __x) { return __isnanf(__x); }
//bool isnan(double __x) { return __isnan(__x); }

bool isgreater(float __x, float __y) {
  return __builtin_isgreater(__x, __y);
}
bool isgreater(double __x, double __y) {
  return __builtin_isgreater(__x, __y);
}
bool isgreaterequal(float __x, float __y) {
  return __builtin_isgreaterequal(__x, __y);
}
bool isgreaterequal(double __x, double __y) {
  return __builtin_isgreaterequal(__x, __y);
}
bool isless(float __x, float __y) {
  return __builtin_isless(__x, __y);
}
bool isless(double __x, double __y) {
  return __builtin_isless(__x, __y);
}
bool islessequal(float __x, float __y) {
  return __builtin_islessequal(__x, __y);
}
bool islessequal(double __x, double __y) {
  return __builtin_islessequal(__x, __y);
}
bool islessgreater(float __x, float __y) {
  return __builtin_islessgreater(__x, __y);
}
bool islessgreater(double __x, double __y) {
  return __builtin_islessgreater(__x, __y);
}
bool isnormal(float __x) { return __builtin_isnormal(__x); }
bool isnormal(double __x) { return __builtin_isnormal(__x); }
bool isunordered(float __x, float __y) {
  return __builtin_isunordered(__x, __y);
}
bool isunordered(double __x, double __y) {
  return __builtin_isunordered(__x, __y);
}
float ldexp(float __arg, int __exp) {
  return ldexpf(__arg, __exp);
}

float log(float __x) { return logf(__x); }
float log10(float __x) { return log10f(__x); }
float modf(float __x, float *__iptr) { return modff(__x, __iptr); }
float pow(float __base, float __exp) {
  return powf(__base, __exp);
}
float pow(float __base, int __iexp) {
  return powif(__base, __iexp);
}
double pow(double __base, int __iexp) {
  return powi(__base, __iexp);
}
/*
bool signbit(float __x) { return __signbitf(__x); }
bool signbit(double __x) { return __signbitd(__x); }
*/
float sin(float __x) { return sinf(__x); }
float sinh(float __x) { return sinhf(__x); }
float sqrt(float __x) { return sqrtf(__x); }
float tan(float __x) { return tanf(__x); }
float tanh(float __x) { return tanhf(__x); }


template <bool __B, class __T = void> struct __clang_cuda_enable_if {};

template <class __T> struct __clang_cuda_enable_if<true, __T> {
  typedef __T type;
};

// Defines an overload of __fn that accepts one integral argument, calls
// __fn((double)x), and returns __retty.
#define __CUDA_CLANG_FN_INTEGER_OVERLOAD_1(__retty, __fn)                      \
  template <typename __T>                                                      \
                                                                    \
      typename __clang_cuda_enable_if<std::numeric_limits<__T>::is_integer,    \
                                      __retty>::type                           \
      __fn(__T __x) {                                                          \
    return ::__fn((double)__x);                                                \
  }

// Defines an overload of __fn that accepts one two arithmetic arguments, calls
// __fn((double)x, (double)y), and returns a double.
//
// Note this is different from OVERLOAD_1, which generates an overload that
// accepts only *integral* arguments.
#define __CUDA_CLANG_FN_INTEGER_OVERLOAD_2(__retty, __fn)                      \
  template <typename __T1, typename __T2>                                      \
  typename __clang_cuda_enable_if<                                  \
      std::numeric_limits<__T1>::is_specialized &&                             \
          std::numeric_limits<__T2>::is_specialized,                           \
      __retty>::type                                                           \
  __fn(__T1 __x, __T2 __y) {                                                   \
    return __fn((double)__x, (double)__y);                                     \
  }

__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(double, acos)
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(double, acosh)
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(double, asin)
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(double, asinh)
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(double, atan)
__CUDA_CLANG_FN_INTEGER_OVERLOAD_2(double, atan2);
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(double, atanh)
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(double, cbrt)
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(double, ceil)
__CUDA_CLANG_FN_INTEGER_OVERLOAD_2(double, copysign);
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(double, cos)
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(double, cosh)
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(double, erf)
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(double, erfc)
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(double, exp)
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(double, exp2)
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(double, expm1)
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(double, fabs)
__CUDA_CLANG_FN_INTEGER_OVERLOAD_2(double, fdim);
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(double, floor)
__CUDA_CLANG_FN_INTEGER_OVERLOAD_2(double, fmax);
__CUDA_CLANG_FN_INTEGER_OVERLOAD_2(double, fmin);
__CUDA_CLANG_FN_INTEGER_OVERLOAD_2(double, fmod);
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(int, fpclassify)
__CUDA_CLANG_FN_INTEGER_OVERLOAD_2(double, hypot);
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(int, ilogb)
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(bool, isfinite)
__CUDA_CLANG_FN_INTEGER_OVERLOAD_2(bool, isgreater);
__CUDA_CLANG_FN_INTEGER_OVERLOAD_2(bool, isgreaterequal);
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(bool, isinf);
__CUDA_CLANG_FN_INTEGER_OVERLOAD_2(bool, isless);
__CUDA_CLANG_FN_INTEGER_OVERLOAD_2(bool, islessequal);
__CUDA_CLANG_FN_INTEGER_OVERLOAD_2(bool, islessgreater);
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(bool, isnan);
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(bool, isnormal)
__CUDA_CLANG_FN_INTEGER_OVERLOAD_2(bool, isunordered);
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(double, lgamma)
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(double, log)
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(double, log10)
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(double, log1p)
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(double, log2)
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(double, logb)
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(long long, llrint)
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(long long, llround)
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(long, lrint)
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(long, lround)
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(double, nearbyint);
__CUDA_CLANG_FN_INTEGER_OVERLOAD_2(double, nextafter);
__CUDA_CLANG_FN_INTEGER_OVERLOAD_2(double, pow);
__CUDA_CLANG_FN_INTEGER_OVERLOAD_2(double, remainder);
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(double, rint);
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(double, round);
//__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(bool, signbit)
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(double, sin)
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(double, sinh)
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(double, sqrt)
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(double, tan)
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(double, tanh)
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(double, tgamma)
__CUDA_CLANG_FN_INTEGER_OVERLOAD_1(double, trunc);

#undef __CUDA_CLANG_FN_INTEGER_OVERLOAD_1
#undef __CUDA_CLANG_FN_INTEGER_OVERLOAD_2

// Overloads for functions that don't match the patterns expected by
// __CUDA_CLANG_FN_INTEGER_OVERLOAD_{1,2}.
template <typename __T1, typename __T2, typename __T3>
typename __clang_cuda_enable_if<
    std::numeric_limits<__T1>::is_specialized &&
        std::numeric_limits<__T2>::is_specialized &&
        std::numeric_limits<__T3>::is_specialized,
    double>::type
fma(__T1 __x, __T2 __y, __T3 __z) {
  return std::fma((double)__x, (double)__y, (double)__z);
}

template <typename __T>
typename __clang_cuda_enable_if<std::numeric_limits<__T>::is_integer,
                                           double>::type
frexp(__T __x, int *__exp) {
  return std::frexp((double)__x, __exp);
}

template <typename __T>
typename __clang_cuda_enable_if<std::numeric_limits<__T>::is_integer,
                                           double>::type
ldexp(__T __x, int __exp) {
  return std::ldexp((double)__x, __exp);
}

template <typename __T1, typename __T2>
typename __clang_cuda_enable_if<
    std::numeric_limits<__T1>::is_specialized &&
        std::numeric_limits<__T2>::is_specialized,
    double>::type
remquo(__T1 __x, __T2 __y, int *__quo) {
  return std::remquo((double)__x, (double)__y, __quo);
}

template <typename __T>
typename __clang_cuda_enable_if<std::numeric_limits<__T>::is_integer,
                                           double>::type
scalbln(__T __x, long __exp) {
  return std::scalbln((double)__x, __exp);
}

template <typename __T>
typename __clang_cuda_enable_if<std::numeric_limits<__T>::is_integer,
                                           double>::type
scalbn(__T __x, int __exp) {
  return std::scalbn((double)__x, __exp);
}

#pragma omp end declare target
#endif // if c++

