//===-------- Interface.h - OpenMP interface ---------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//


#if defined(__AMDGCN__) || defined(__NVPTX__)
#else

extern "C" {

///}
///  Helper functions for high performance cross team reductions.
///
///    Name decoder:  __kmpc_xteamr_<dtype>_<waves>x<warpsize>
///       <waves> number of warps/waves in team
///       <warpsize> 32 or 64
///       example: __kmpc_xteam_f_16x64 is the helper function for data
///                type float with fixed teamsize of 1024 threads.
///    There are 48(6 X 8) helper functions.
///    6 configurations of teamsize are currently created.
///    The most performant configs use max teamsize 1024: 16x64 and 32x32.
///    Currently the Other confis are 8x64, 4x64, 16x32, and 8x32.
///    8 data types available: double, float, int, uint, long, ulong,
///    double  _Complex, and float _complex
///    All xteam helper functions have these 7 args:
///      arg1: the thread local reduction value
///      arg2: pointer to where result is written
///      arg3: global array of team values for this reduction instance
///      arg4: atomic counter of completed teams for this reduction instance
///      arg5: void function pointer of  pair reduction function,
///            (e.g. sum(&a,b),min(&a,b),max(&a,b)
///      arg6: equivalent (to arg5) void function pointer of pair reduction
///            function on LDS memory, (e.g. sum(&a,&b),min(&a,&b),max(&a,&b)
///      arg7: Initializing value for the reduction type
//
#define _RF_LDS 
void __kmpc_xteamr_d_16x64(double v, double *r_ptr, double *tvals,
                           uint32_t *td_ptr, void (*_rf)(double *, double),
                           void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *),
                           double iv)
{}
void __kmpc_xteamr_f_16x64(float v, float *r_ptr, float *tvals,
                           uint32_t *td_ptr, void (*_rf)(float *, float),
                           void (*_rf_lds)(_RF_LDS float *, _RF_LDS float *),
                           float iv)
{}
void __kmpc_xteamr_cd_16x64(double _Complex v, double _Complex *r_ptr,
                            double _Complex *tvals, uint32_t *td_ptr,
                            void (*_rf)(double _Complex *, double _Complex),
                            void (*_rf_lds)(_RF_LDS double _Complex *,
                                            _RF_LDS double _Complex *),
                            double _Complex iv)
{}
void __kmpc_xteamr_cf_16x64(float _Complex v, float _Complex *r_ptr,
                            float _Complex *tvals, uint32_t *td_ptr,
                            void (*_rf)(float _Complex *, float _Complex),
                            void (*_rf_lds)(_RF_LDS float _Complex *,
                                            _RF_LDS float _Complex *),
                            float _Complex iv)
{}
void __kmpc_xteamr_i_16x64(int v, int *r_ptr, int *tvals, uint32_t *td_ptr,
                           void (*_rf)(int *, int),
                           void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *),
                           int iv)
{}
void __kmpc_xteamr_ui_16x64(uint32_t v, uint32_t *r_ptr, uint32_t *tvals,
                            uint32_t *td_ptr, void (*_rf)(uint32_t *, uint32_t),
                            void (*_rf_lds)(_RF_LDS uint32_t *,
                                            _RF_LDS uint32_t *),
                            uint32_t iv)
{}
void __kmpc_xteamr_l_16x64(long v, long *r_ptr, long *tvals, uint32_t *td_ptr,
                           void (*_rf)(long *, long),
                           void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *),
                           long iv)
{}
void __kmpc_xteamr_ul_16x64(uint64_t v, uint64_t *r_ptr, uint64_t *tvals,
                            uint32_t *td_ptr, void (*_rf)(uint64_t *, uint64_t),
                            void (*_rf_lds)(_RF_LDS uint64_t *,
                                            _RF_LDS uint64_t *),
                            uint64_t iv)
{}
void __kmpc_xteamr_d_8x64(double v, double *r_ptr, double *tvals,
                          uint32_t *td_ptr, void (*_rf)(double *, double),
                          void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *),
                          double iv)
{}
void __kmpc_xteamr_f_8x64(float v, float *r_ptr, float *tvals, uint32_t *td_ptr,
                          void (*_rf)(float *, float),
                          void (*_rf_lds)(_RF_LDS float *, _RF_LDS float *),
                          float iv)
{}
void __kmpc_xteamr_cd_8x64(double _Complex v, double _Complex *r_ptr,
                           double _Complex *tvals, uint32_t *td_ptr,
                           void (*_rf)(double _Complex *, double _Complex),
                           void (*_rf_lds)(_RF_LDS double _Complex *,
                                           _RF_LDS double _Complex *),
                           double _Complex iv)
{}
void __kmpc_xteamr_cf_8x64(float _Complex v, float _Complex *r_ptr,
                           float _Complex *tvals, uint32_t *td_ptr,
                           void (*_rf)(float _Complex *, float _Complex),
                           void (*_rf_lds)(_RF_LDS float _Complex *,
                                           _RF_LDS float _Complex *),
                           float _Complex iv)
{}
void __kmpc_xteamr_i_8x64(int v, int *r_ptr, int *tvals, uint32_t *td_ptr,
                          void (*_rf)(int *, int),
                          void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *),
                          int iv)
{}
void __kmpc_xteamr_ui_8x64(uint32_t v, uint32_t *r_ptr, uint32_t *tvals,
                           uint32_t *td_ptr, void (*_rf)(uint32_t *, uint32_t),
                           void (*_rf_lds)(_RF_LDS uint32_t *,
                                           _RF_LDS uint32_t *),
                           uint32_t iv)
{}
void __kmpc_xteamr_l_8x64(long v, long *r_ptr, long *tvals, uint32_t *td_ptr,
                          void (*_rf)(long *, long),
                          void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *),
                          long iv)
{}
void __kmpc_xteamr_ul_8x64(uint64_t v, uint64_t *r_ptr, uint64_t *tvals,
                           uint32_t *td_ptr, void (*_rf)(uint64_t *, uint64_t),
                           void (*_rf_lds)(_RF_LDS uint64_t *,
                                           _RF_LDS uint64_t *),
                           uint64_t iv)
{}
void __kmpc_xteamr_d_4x64(double v, double *r_ptr, double *tvals,
                          uint32_t *td_ptr, void (*_rf)(double *, double),
                          void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *),
                          double iv)
{}
void __kmpc_xteamr_f_4x64(float v, float *r_ptr, float *tvals, uint32_t *td_ptr,
                          void (*_rf)(float *, float),
                          void (*_rf_lds)(_RF_LDS float *, _RF_LDS float *),
                          float iv)
{}
void __kmpc_xteamr_i_4x64(int v, int *r_ptr, int *tvals, uint32_t *td_ptr,
                          void (*_rf)(int *, int),
                          void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *),
                          int iv)
{}
void __kmpc_xteamr_ui_4x64(uint32_t v, uint32_t *r_ptr, uint32_t *tvals,
                           uint32_t *td_ptr, void (*_rf)(uint32_t *, uint32_t),
                           void (*_rf_lds)(_RF_LDS uint32_t *,
                                           _RF_LDS uint32_t *),
                           uint32_t iv)
{}
void __kmpc_xteamr_l_4x64(long v, long *r_ptr, long *tvals, uint32_t *td_ptr,
                          void (*_rf)(long *, long),
                          void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *),
                          long iv)
{}
void __kmpc_xteamr_ul_4x64(uint64_t v, uint64_t *r_ptr, uint64_t *tvals,
                           uint32_t *td_ptr, void (*_rf)(uint64_t *, uint64_t),
                           void (*_rf_lds)(_RF_LDS uint64_t *,
                                           _RF_LDS uint64_t *),
                           uint64_t iv)
{}
void __kmpc_xteamr_d_32x32(double v, double *r_ptr, double *tvals,
                           uint32_t *td_ptr, void (*_rf)(double *, double),
                           void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *),
                           double iv)

{}
void __kmpc_xteamr_f_32x32(float v, float *r_ptr, float *tvals,
                           uint32_t *td_ptr, void (*_rf)(float *, float),
                           void (*_rf_lds)(_RF_LDS float *, _RF_LDS float *),
                           float iv)

{}
void __kmpc_xteamr_i_32x32(int v, int *r_ptr, int *tvals, uint32_t *td_ptr,
                           void (*_rf)(int *, int),
                           void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *),
                           int iv)

{}
void __kmpc_xteamr_ui_32x32(uint32_t v, uint32_t *r_ptr, uint32_t *tvals,
                            uint32_t *td_ptr, void (*_rf)(uint32_t *, uint32_t),
                            void (*_rf_lds)(_RF_LDS uint32_t *,
                                            _RF_LDS uint32_t *),
                            uint32_t iv)
{}
void __kmpc_xteamr_l_32x32(long v, long *r_ptr, long *tvals, uint32_t *td_ptr,
                           void (*_rf)(long *, long),
                           void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *),
                           long iv)
{}
void __kmpc_xteamr_ul_32x32(uint64_t v, uint64_t *r_ptr, uint64_t *tvals,
                            uint32_t *td_ptr, void (*_rf)(uint64_t *, uint64_t),
                            void (*_rf_lds)(_RF_LDS uint64_t *,
                                            _RF_LDS uint64_t *),
                            uint64_t iv)
{}
void __kmpc_xteamr_d_16x32(double v, double *r_ptr, double *tvals,
                           uint32_t *td_ptr, void (*_rf)(double *, double),
                           void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *),
                           double iv)

{}
void __kmpc_xteamr_f_16x32(float v, float *r_ptr, float *tvals,
                           uint32_t *td_ptr, void (*_rf)(float *, float),
                           void (*_rf_lds)(_RF_LDS float *, _RF_LDS float *),
                           float iv)

{}
void __kmpc_xteamr_i_16x32(int v, int *r_ptr, int *tvals, uint32_t *td_ptr,
                           void (*_rf)(int *, int),
                           void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *),
                           int iv)
{}
void __kmpc_xteamr_ui_16x32(uint32_t v, uint32_t *r_ptr, uint32_t *tvals,
                            uint32_t *td_ptr, void (*_rf)(uint32_t *, uint32_t),
                            void (*_rf_lds)(_RF_LDS uint32_t *,
                                            _RF_LDS uint32_t *),
                            uint32_t iv)
{}
void __kmpc_xteamr_l_16x32(long v, long *r_ptr, long *tvals, uint32_t *td_ptr,
                           void (*_rf)(long *, long),
                           void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *),
                           long iv)
{}
void __kmpc_xteamr_ul_16x32(uint64_t v, uint64_t *r_ptr, uint64_t *tvals,
                            uint32_t *td_ptr, void (*_rf)(uint64_t *, uint64_t),
                            void (*_rf_lds)(_RF_LDS uint64_t *,
                                            _RF_LDS uint64_t *),
                            uint64_t iv)
{}
void __kmpc_xteamr_d_8x32(double v, double *r_ptr, double *tvals,
                          uint32_t *td_ptr, void (*_rf)(double *, double),
                          void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *),
                          double iv)
{}
void __kmpc_xteamr_f_8x32(float v, float *r_ptr, float *tvals, uint32_t *td_ptr,
                          void (*_rf)(float *, float),
                          void (*_rf_lds)(_RF_LDS float *, _RF_LDS float *),
                          float iv)
{}
void __kmpc_xteamr_i_8x32(int v, int *r_ptr, int *tvals, uint32_t *td_ptr,
                          void (*_rf)(int *, int),
                          void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *),
                          int iv)
{}
void __kmpc_xteamr_ui_8x32(uint32_t v, uint32_t *r_ptr, uint32_t *tvals,
                           uint32_t *td_ptr, void (*_rf)(uint32_t *, uint32_t),
                           void (*_rf_lds)(_RF_LDS uint32_t *,
                                           _RF_LDS uint32_t *),
                           uint32_t iv)
{}
void __kmpc_xteamr_l_8x32(long v, long *r_ptr, long *tvals, uint32_t *td_ptr,
                          void (*_rf)(long *, long),
                          void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *),
                          long iv)
{}
void __kmpc_xteamr_ul_8x32(uint64_t v, uint64_t *r_ptr, uint64_t *tvals,
                           uint32_t *td_ptr, void (*_rf)(uint64_t *, uint64_t),
                           void (*_rf_lds)(_RF_LDS uint64_t *,
                                           _RF_LDS uint64_t *),
                           uint64_t iv)
{}

///  Builtin pair reduction functions.
///    These become function pointers for arg5 and arg6 of xteamr above.
///    Each pair reduction function must have two variants to support xteamr.
///    The 1st is for TLS memory and the 2nd is for LDS (scratchpad) memory.
///    These are defined in Rfuns.cpp.  User defined reductions require
///    that Clang codegen generate these functions.
void __kmpc_rfun_sum_d(double *val, double otherval)
{}
void __kmpc_rfun_sum_lds_d(_RF_LDS double *val, _RF_LDS double *otherval)
{}
void __kmpc_rfun_sum_f(float *val, float otherval)
{}
void __kmpc_rfun_sum_lds_f(_RF_LDS float *val, _RF_LDS float *otherval)
{}
void __kmpc_rfun_sum_cd(double _Complex *val, double _Complex otherval)
{}
void __kmpc_rfun_sum_lds_cd(_RF_LDS double _Complex *val,
                           _RF_LDS double _Complex *otherval)
{}
void __kmpc_rfun_sum_cf(float _Complex *val, float _Complex otherval)
{}
void __kmpc_rfun_sum_lds_cf(_RF_LDS float _Complex *val,
                           _RF_LDS float _Complex *otherval)
{}
void __kmpc_rfun_sum_i(int *val, int otherval)
{}
void __kmpc_rfun_sum_lds_i(_RF_LDS int *val, _RF_LDS int *otherval)
{}
void __kmpc_rfun_sum_ui(unsigned int *val, unsigned int otherval)
{}
void __kmpc_rfun_sum_lds_ui(_RF_LDS unsigned int *val,
                            _RF_LDS unsigned int *otherval)
{}
void __kmpc_rfun_sum_l(long *val, long otherval)
{}
void __kmpc_rfun_sum_lds_l(_RF_LDS long *val, _RF_LDS long *otherval)
{}
void __kmpc_rfun_sum_ul(unsigned long *val, unsigned long otherval)
{}
void __kmpc_rfun_sum_lds_ul(_RF_LDS unsigned long *val,
                            _RF_LDS unsigned long *otherval)
{}
/// Complex variables have no compare, so no min or max for cf and cd.
void __kmpc_rfun_min_d(double *val, double otherval)
{}
void __kmpc_rfun_min_lds_d(_RF_LDS double *val, _RF_LDS double *otherval)
{}
void __kmpc_rfun_min_f(float *val, float otherval)
{}
void __kmpc_rfun_min_lds_f(_RF_LDS float *val, _RF_LDS float *otherval)
{}
void __kmpc_rfun_min_i(int *val, int otherval)
{}
void __kmpc_rfun_min_lds_i(_RF_LDS int *val, _RF_LDS int *otherval)
{}
void __kmpc_rfun_min_ui(unsigned int *val, unsigned int otherval)
{}
void __kmpc_rfun_min_lds_ui(_RF_LDS unsigned int *val,
                            _RF_LDS unsigned int *otherval)
{}
void __kmpc_rfun_min_l(long *val, long otherval)
{}
void __kmpc_rfun_min_lds_l(_RF_LDS long *val, _RF_LDS long *otherval)
{}
void __kmpc_rfun_min_ul(unsigned long *val, unsigned long otherval)
{}
void __kmpc_rfun_min_lds_ul(_RF_LDS unsigned long *val,
                            _RF_LDS unsigned long *otherval)
{}
void __kmpc_rfun_max_d(double *val, double otherval)
{}
void __kmpc_rfun_max_lds_d(_RF_LDS double *val, _RF_LDS double *otherval)
{}
void __kmpc_rfun_max_f(float *val, float otherval)
{}
void __kmpc_rfun_max_lds_f(_RF_LDS float *val, _RF_LDS float *otherval)
{}
void __kmpc_rfun_max_i(int *val, int otherval)
{}
void __kmpc_rfun_max_lds_i(_RF_LDS int *val, _RF_LDS int *otherval)
{}
void __kmpc_rfun_max_ui(unsigned int *val, unsigned int otherval)
{}
void __kmpc_rfun_max_lds_ui(_RF_LDS unsigned int *val,
                            _RF_LDS unsigned int *otherval)
{}
void __kmpc_rfun_max_l(long *val, long otherval)
{}
void __kmpc_rfun_max_lds_l(_RF_LDS long *val, _RF_LDS long *otherval)
{}
void __kmpc_rfun_max_ul(unsigned long *val, unsigned long otherval)
{}
void __kmpc_rfun_max_lds_ul(_RF_LDS unsigned long *val,
                            _RF_LDS unsigned long *otherval)
{}
#undef _RF_LDS
}
#endif
