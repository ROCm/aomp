//===----------------------------------------------------------------------===//
//
// This file contains cpu versions of the cross team helper functions
// to test the fortran interfaces to c helper functions
// The device versions for target regions are in another test directory
//
//===----------------------------------------------------------------------===//

#include <cstdio>

/// #define _EXT_ATTR extern "C" __attribute__((flatten, always_inline)) void
#define _EXT_ATTR extern "C" void
#define _CD double _Complex
#define _CF float _Complex
#define _UI unsigned int
#define _UL unsigned long
// #define _LDS volatile __attribute__((address_space(3)))
#define _LDS
// #define _UN_ unsigned
#define _UN_ 

_EXT_ATTR
__kmpc_xteamr_i_4x64(int v, int *r_p, int *tvs, _UN_ int *td,
  void (*rf)(int *, int), void (*rflds)(_LDS int *, _LDS int *),
  const int rnv, const _UN_ long int k, const _UN_ int nt) {
//  printf("%ld  __kmpc_xteamr_i_4x64 called with v=%d f:%p\n",k,v,rf);
//  _xteam_reduction<int, 4, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
  rf(r_p, v);
}

_EXT_ATTR
__kmpc_xteamr_d_4x64(double v, double *r_p, double *tvs, _UN_ int *td,
  void (*rf)(double *, double), void (*rflds)(_LDS double *, _LDS double *),
  const double rnv, const _UN_ long int k, const _UN_ int nt) {
  // _xteam_reduction<double, 4, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
  rf(r_p, v);
}

#undef _CD
#undef _CF
#undef _UI
#undef _UL
#undef _LDS

// Built-in pair reduction functions used as function pointers for
// cross team reduction functions.

// #define _RF_LDS volatile __attribute__((address_space(3)))
#define _RF_LDS

_EXT_ATTR __kmpc_rfun_sum_i(int *val, int otherval) {
  // printf(" rfun_sum_i called with value %d and %d result=%d val:%p \n",otherval, *val, *val+otherval,val);
  *val += otherval; 
}
_EXT_ATTR __kmpc_rfun_sum_lds_i(_RF_LDS int *val, _RF_LDS int *otherval) {
  *val += *otherval;
}
_EXT_ATTR __kmpc_rfun_sum_d(double *val, double otherval) { *val += otherval; }
_EXT_ATTR __kmpc_rfun_sum_lds_d(_RF_LDS double *val, _RF_LDS double *otherval) {
  *val += *otherval;
}

#undef _EXT_ATTR
#undef _RF_LDS

#pragma omp end declare target
