//===--------- libm/libm.c ------------------------------------------------===//
//
//                The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// If the library needs to destinguish betwen different targets,
// target specific macros for that GPU can be used.
// For nvptx use __NVPTX__.  For amdgcn, use __AMDGCN__.
// Example:
//   #ifdef __AMDGCN__ && (__AMDGCN__ == 1000)
//     double fast_sqrt(double __a) { ... }
//   #endif

#ifdef __AMDGCN__
#include "libm-amdgcn.c"
#endif

#ifdef __NVPTX__
#include "libm-nvptx.c"
#endif
