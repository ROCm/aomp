//===--------- libm/libm-amdgcn.cpp ---------------------------------------===//
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
#include <stdint.h>
#include "libm-amdgcn.h"
#pragma omp declare target

/// Do not provide the double overload because the base functions
/// are defined in libm-amdgcn.c.  Only provide non-base variants.

int abs(int x)
{
    int sgn = x >> (sizeof(int) * CHAR_BIT - 1);
    return (x ^ sgn) - sgn;
}
long labs(long x)
{
    long sgn = x >> (sizeof(long) * CHAR_BIT - 1);
    return (x ^ sgn) - sgn;
}
long long llabs(long long x)
{
    long long sgn = x >> (sizeof(long long) * CHAR_BIT - 1);
    return (x ^ sgn) - sgn;
}

// Template helpers
template<bool __B, class __T = void>
struct __amdgcn_enable_if {};

template <class __T> struct __amdgcn_enable_if<true, __T> {
  typedef __T type;
};

#define __AMDGCN_OVERLOAD1(__retty, __fn)                                      \
  template <typename __T>                                                      \
      typename __amdgcn_enable_if<std::numeric_limits<__T>::is_integer,        \
                                      __retty>::type                           \
      __fn(__T __x) {                                                          \
    return ::__fn((double)__x);                                                \
  }

#define __AMDGCN_OVERLOAD2(__retty, __fn)                                      \
  template <typename __T1, typename __T2>                                      \
       typename __amdgcn_enable_if<                                            \
      std::numeric_limits<__T1>::is_specialized &&                             \
          std::numeric_limits<__T2>::is_specialized,                           \
      __retty>::type                                                           \
  __fn(__T1 __x, __T2 __y) {                                                   \
    return __fn((double)__x, (double)__y);                                     \
  }
#define __AMDGCN_OVERLOAD3(__retty, __fn)                                      \
  template <typename __T1, typename __T2, typename __T3>                       \
             typename __amdgcn_enable_if<                                      \
      std::numeric_limits<__T1>::is_specialized &&                             \
          std::numeric_limits<__T2>::is_specialized &&                         \
          std::numeric_limits<__T3>::is_specialized,                           \
      __retty>::type                                                           \
  __fn(__T1 __x, __T2 __y, __T3 __z) {                                         \
    return __fn((double)__x, (double)__y, (double)__z);                        \
  }
#define __DEF_FUN1(retty, func) \
inline \
float func(float x) \
{ \
  return func##f(x); \
} \
__AMDGCN_OVERLOAD1(retty, func)

// Define cmath functions with float argument and returns retty.
#define __DEF_FUNI(retty, func) \
inline \
retty func(float x) \
{ \
  return func##f(x); \
} \
__AMDGCN_OVERLOAD1(retty, func)

// define cmath functions with two float arguments.
#define __DEF_FUN2(retty, func) \
inline \
float func(float x, float y) \
{ \
  return func##f(x, y); \
} \
__AMDGCN_OVERLOAD2(retty, func)

// define cmath functions with three float arguments.
#define __DEF_FUN3(retty, func) \
inline \
float func(float x, float y, float z) \
{ \
  return func##f(x, y, z); \
} \
__AMDGCN_OVERLOAD3(retty, func)
// cmath functions
long abs(long x) { return labs(x); }
long long abs(long long x) { return llabs(x); }
__DEF_FUN1(double, acos)
__DEF_FUN1(double, acosh)
__DEF_FUN1(double, asin)
__DEF_FUN1(double, asinh)
__DEF_FUN1(double, atan)
__DEF_FUN2(double, atan2);
__DEF_FUN1(double, atanh)
__DEF_FUN1(double, cbrt)
__DEF_FUN1(double, ceil)
__DEF_FUN2(double, copysign);
__DEF_FUN1(double, cos)
__DEF_FUN1(double, cosh)
__DEF_FUN1(double, erf)
__DEF_FUN1(double, erfc)
__DEF_FUN1(double, exp)
__DEF_FUN1(double, exp2)
__DEF_FUN1(double, expm1)
__DEF_FUN1(double, fabs)
__DEF_FUN2(double, fdim);
__DEF_FUN1(double, floor)
__DEF_FUN3(double, fma)
__DEF_FUN2(double, fmax);
__DEF_FUN2(double, fmin);
__DEF_FUN2(double, fmod);
//__AMDGCN_OVERLOAD1(int, fpclassify)
__DEF_FUN2(double, hypot);
__DEF_FUNI(int, ilogb)
//__AMDGCN_OVERLOAD1(bool, isfinite)
__AMDGCN_OVERLOAD2(bool, isgreater);
__AMDGCN_OVERLOAD2(bool, isgreaterequal);
//__AMDGCN_OVERLOAD1(bool, isinf);
__AMDGCN_OVERLOAD2(bool, isless);
__AMDGCN_OVERLOAD2(bool, islessequal);
__AMDGCN_OVERLOAD2(bool, islessgreater);
//__AMDGCN_OVERLOAD1(bool, isnan);
//__AMDGCN_OVERLOAD1(bool, isnormal)
__AMDGCN_OVERLOAD2(bool, isunordered);
__DEF_FUN1(double, lgamma)
__DEF_FUN1(double, log)
__DEF_FUN1(double, log10)
__DEF_FUN1(double, log1p)
__DEF_FUN1(double, log2)
__DEF_FUN1(double, logb)
__DEF_FUNI(long long, llrint)
__DEF_FUNI(long long, llround)
__DEF_FUNI(long, lrint)
__DEF_FUNI(long, lround)
__DEF_FUN1(double, nearbyint);
__DEF_FUN2(double, nextafter);
__DEF_FUN2(double, pow);
__DEF_FUN2(double, remainder);
__DEF_FUN1(double, rint);
__DEF_FUN1(double, round);
//__AMDGCN_OVERLOAD1(bool, signbit)
__DEF_FUN1(double, sin)
__DEF_FUN1(double, sinh)
__DEF_FUN1(double, sqrt)
__DEF_FUN1(double, tan)
__DEF_FUN1(double, tanh)
__DEF_FUN1(double, tgamma)
__DEF_FUN1(double, trunc);

#pragma omp end declare target
