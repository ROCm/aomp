# 1 "c_timers.c"
# 1 "<built-in>" 1
# 1 "<built-in>" 3
# 671 "<built-in>" 3
# 1 "<command line>" 1
# 1 "<built-in>" 2
# 1 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/openmp_wrappers/__clang_openmp_device_functions.h" 1 3
# 52 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/openmp_wrappers/__clang_openmp_device_functions.h" 3
#pragma omp begin declare variant match( device = {arch(amdgcn)}, implementation = {extension(match_any)})







# 1 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/stdint.h" 1 3
# 52 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/stdint.h" 3
# 1 "/usr/include/stdint.h" 1 3 4
# 26 "/usr/include/stdint.h" 3 4
# 1 "/usr/include/bits/libc-header-start.h" 1 3 4
# 33 "/usr/include/bits/libc-header-start.h" 3 4
# 1 "/usr/include/features.h" 1 3 4
# 406 "/usr/include/features.h" 3 4
# 1 "/usr/include/stdc-predef.h" 1 3 4
# 407 "/usr/include/features.h" 2 3 4
# 428 "/usr/include/features.h" 3 4
# 1 "/usr/include/sys/cdefs.h" 1 3 4
# 442 "/usr/include/sys/cdefs.h" 3 4
# 1 "/usr/include/bits/wordsize.h" 1 3 4
# 443 "/usr/include/sys/cdefs.h" 2 3 4
# 1 "/usr/include/bits/long-double.h" 1 3 4
# 444 "/usr/include/sys/cdefs.h" 2 3 4
# 429 "/usr/include/features.h" 2 3 4
# 452 "/usr/include/features.h" 3 4
# 1 "/usr/include/gnu/stubs.h" 1 3 4
# 10 "/usr/include/gnu/stubs.h" 3 4
# 1 "/usr/include/gnu/stubs-64.h" 1 3 4
# 11 "/usr/include/gnu/stubs.h" 2 3 4
# 453 "/usr/include/features.h" 2 3 4
# 34 "/usr/include/bits/libc-header-start.h" 2 3 4
# 27 "/usr/include/stdint.h" 2 3 4
# 1 "/usr/include/bits/types.h" 1 3 4
# 27 "/usr/include/bits/types.h" 3 4
# 1 "/usr/include/bits/wordsize.h" 1 3 4
# 28 "/usr/include/bits/types.h" 2 3 4


typedef unsigned char __u_char;
typedef unsigned short int __u_short;
typedef unsigned int __u_int;
typedef unsigned long int __u_long;


typedef signed char __int8_t;
typedef unsigned char __uint8_t;
typedef signed short int __int16_t;
typedef unsigned short int __uint16_t;
typedef signed int __int32_t;
typedef unsigned int __uint32_t;

typedef signed long int __int64_t;
typedef unsigned long int __uint64_t;






typedef __int8_t __int_least8_t;
typedef __uint8_t __uint_least8_t;
typedef __int16_t __int_least16_t;
typedef __uint16_t __uint_least16_t;
typedef __int32_t __int_least32_t;
typedef __uint32_t __uint_least32_t;
typedef __int64_t __int_least64_t;
typedef __uint64_t __uint_least64_t;



typedef long int __quad_t;
typedef unsigned long int __u_quad_t;







typedef long int __intmax_t;
typedef unsigned long int __uintmax_t;
# 140 "/usr/include/bits/types.h" 3 4
# 1 "/usr/include/bits/typesizes.h" 1 3 4
# 141 "/usr/include/bits/types.h" 2 3 4


typedef unsigned long int __dev_t;
typedef unsigned int __uid_t;
typedef unsigned int __gid_t;
typedef unsigned long int __ino_t;
typedef unsigned long int __ino64_t;
typedef unsigned int __mode_t;
typedef unsigned long int __nlink_t;
typedef long int __off_t;
typedef long int __off64_t;
typedef int __pid_t;
typedef struct { int __val[2]; } __fsid_t;
typedef long int __clock_t;
typedef unsigned long int __rlim_t;
typedef unsigned long int __rlim64_t;
typedef unsigned int __id_t;
typedef long int __time_t;
typedef unsigned int __useconds_t;
typedef long int __suseconds_t;

typedef int __daddr_t;
typedef int __key_t;


typedef int __clockid_t;


typedef void * __timer_t;


typedef long int __blksize_t;




typedef long int __blkcnt_t;
typedef long int __blkcnt64_t;


typedef unsigned long int __fsblkcnt_t;
typedef unsigned long int __fsblkcnt64_t;


typedef unsigned long int __fsfilcnt_t;
typedef unsigned long int __fsfilcnt64_t;


typedef long int __fsword_t;

typedef long int __ssize_t;


typedef long int __syscall_slong_t;

typedef unsigned long int __syscall_ulong_t;



typedef __off64_t __loff_t;
typedef char *__caddr_t;


typedef long int __intptr_t;


typedef unsigned int __socklen_t;




typedef int __sig_atomic_t;
# 28 "/usr/include/stdint.h" 2 3 4
# 1 "/usr/include/bits/wchar.h" 1 3 4
# 29 "/usr/include/stdint.h" 2 3 4
# 1 "/usr/include/bits/wordsize.h" 1 3 4
# 30 "/usr/include/stdint.h" 2 3 4




# 1 "/usr/include/bits/stdint-intn.h" 1 3 4
# 24 "/usr/include/bits/stdint-intn.h" 3 4
typedef __int8_t int8_t;
typedef __int16_t int16_t;
typedef __int32_t int32_t;
typedef __int64_t int64_t;
# 35 "/usr/include/stdint.h" 2 3 4


# 1 "/usr/include/bits/stdint-uintn.h" 1 3 4
# 24 "/usr/include/bits/stdint-uintn.h" 3 4
typedef __uint8_t uint8_t;
typedef __uint16_t uint16_t;
typedef __uint32_t uint32_t;
typedef __uint64_t uint64_t;
# 38 "/usr/include/stdint.h" 2 3 4





typedef __int_least8_t int_least8_t;
typedef __int_least16_t int_least16_t;
typedef __int_least32_t int_least32_t;
typedef __int_least64_t int_least64_t;


typedef __uint_least8_t uint_least8_t;
typedef __uint_least16_t uint_least16_t;
typedef __uint_least32_t uint_least32_t;
typedef __uint_least64_t uint_least64_t;





typedef signed char int_fast8_t;

typedef long int int_fast16_t;
typedef long int int_fast32_t;
typedef long int int_fast64_t;
# 71 "/usr/include/stdint.h" 3 4
typedef unsigned char uint_fast8_t;

typedef unsigned long int uint_fast16_t;
typedef unsigned long int uint_fast32_t;
typedef unsigned long int uint_fast64_t;
# 87 "/usr/include/stdint.h" 3 4
typedef long int intptr_t;


typedef unsigned long int uintptr_t;
# 101 "/usr/include/stdint.h" 3 4
typedef __intmax_t intmax_t;
typedef __uintmax_t uintmax_t;
# 53 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/stdint.h" 2 3
# 60 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/openmp_wrappers/__clang_openmp_device_functions.h" 2 3
# 73 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/openmp_wrappers/__clang_openmp_device_functions.h" 3
# 1 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/__clang_hip_libdevice_declares.h" 1 3
# 18 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/__clang_hip_libdevice_declares.h" 3
__attribute__((device)) __attribute__((const)) float __ocml_acos_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_acosh_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_asin_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_asinh_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_atan2_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_atan_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_atanh_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_cbrt_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_ceil_f32(float);
__attribute__((device)) __attribute__((const)) __attribute__((device)) float __ocml_copysign_f32(float,
                                                                       float);
__attribute__((device)) float __ocml_cos_f32(float);
__attribute__((device)) float __ocml_native_cos_f32(float);
__attribute__((device)) __attribute__((pure)) __attribute__((device)) float __ocml_cosh_f32(float);
__attribute__((device)) float __ocml_cospi_f32(float);
__attribute__((device)) float __ocml_i0_f32(float);
__attribute__((device)) float __ocml_i1_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_erfc_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_erfcinv_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_erfcx_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_erf_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_erfinv_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_exp10_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_native_exp10_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_exp2_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_exp_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_native_exp_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_expm1_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_fabs_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_fdim_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_floor_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_fma_f32(float, float, float);
__attribute__((device)) __attribute__((const)) float __ocml_fmax_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_fmin_f32(float, float);
__attribute__((device)) __attribute__((const)) __attribute__((device)) float __ocml_fmod_f32(float,
                                                                   float);
__attribute__((device)) float __ocml_frexp_f32(float,
                                  __attribute__((address_space(5))) int *);
__attribute__((device)) __attribute__((const)) float __ocml_hypot_f32(float, float);
__attribute__((device)) __attribute__((const)) int __ocml_ilogb_f32(float);
__attribute__((device)) __attribute__((const)) int __ocml_isfinite_f32(float);
__attribute__((device)) __attribute__((const)) int __ocml_isinf_f32(float);
__attribute__((device)) __attribute__((const)) int __ocml_isnan_f32(float);
__attribute__((device)) float __ocml_j0_f32(float);
__attribute__((device)) float __ocml_j1_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_ldexp_f32(float, int);
__attribute__((device)) float __ocml_lgamma_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_log10_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_native_log10_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_log1p_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_log2_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_native_log2_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_logb_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_log_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_native_log_f32(float);
__attribute__((device)) float __ocml_modf_f32(float,
                                 __attribute__((address_space(5))) float *);
__attribute__((device)) __attribute__((const)) float __ocml_nearbyint_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_nextafter_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_len3_f32(float, float, float);
__attribute__((device)) __attribute__((const)) float __ocml_len4_f32(float, float, float,
                                                        float);
__attribute__((device)) __attribute__((pure)) float __ocml_ncdf_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_ncdfinv_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_pow_f32(float, float);
__attribute__((device)) __attribute__((pure)) float __ocml_pown_f32(float, int);
__attribute__((device)) __attribute__((pure)) float __ocml_rcbrt_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_remainder_f32(float, float);
__attribute__((device)) float __ocml_remquo_f32(float, float,
                                   __attribute__((address_space(5))) int *);
__attribute__((device)) __attribute__((const)) float __ocml_rhypot_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_rint_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_rlen3_f32(float, float, float);
__attribute__((device)) __attribute__((const)) float __ocml_rlen4_f32(float, float, float,
                                                         float);
__attribute__((device)) __attribute__((const)) float __ocml_round_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_rsqrt_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_scalb_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_scalbn_f32(float, int);
__attribute__((device)) __attribute__((const)) int __ocml_signbit_f32(float);
__attribute__((device)) float __ocml_sincos_f32(float,
                                   __attribute__((address_space(5))) float *);
__attribute__((device)) float __ocml_sincospi_f32(float,
                                     __attribute__((address_space(5))) float *);
__attribute__((device)) float __ocml_sin_f32(float);
__attribute__((device)) float __ocml_native_sin_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_sinh_f32(float);
__attribute__((device)) float __ocml_sinpi_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_sqrt_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_native_sqrt_f32(float);
__attribute__((device)) float __ocml_tan_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_tanh_f32(float);
__attribute__((device)) float __ocml_tgamma_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_trunc_f32(float);
__attribute__((device)) float __ocml_y0_f32(float);
__attribute__((device)) float __ocml_y1_f32(float);


__attribute__((device)) __attribute__((const)) float __ocml_add_rte_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_add_rtn_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_add_rtp_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_add_rtz_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_sub_rte_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_sub_rtn_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_sub_rtp_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_sub_rtz_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_mul_rte_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_mul_rtn_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_mul_rtp_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_mul_rtz_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_div_rte_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_div_rtn_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_div_rtp_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_div_rtz_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_sqrt_rte_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_sqrt_rtn_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_sqrt_rtp_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_sqrt_rtz_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_fma_rte_f32(float, float, float);
__attribute__((device)) __attribute__((const)) float __ocml_fma_rtn_f32(float, float, float);
__attribute__((device)) __attribute__((const)) float __ocml_fma_rtp_f32(float, float, float);
__attribute__((device)) __attribute__((const)) float __ocml_fma_rtz_f32(float, float, float);

__attribute__((device)) __attribute__((const)) float
__llvm_amdgcn_cos_f32(float) __asm("llvm.amdgcn.cos.f32");
__attribute__((device)) __attribute__((const)) float
__llvm_amdgcn_rcp_f32(float) __asm("llvm.amdgcn.rcp.f32");
__attribute__((device)) __attribute__((const)) float
__llvm_amdgcn_rsq_f32(float) __asm("llvm.amdgcn.rsq.f32");
__attribute__((device)) __attribute__((const)) float
__llvm_amdgcn_sin_f32(float) __asm("llvm.amdgcn.sin.f32");




__attribute__((device)) __attribute__((const)) double __ocml_acos_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_acosh_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_asin_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_asinh_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_atan2_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_atan_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_atanh_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_cbrt_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_ceil_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_copysign_f64(double, double);
__attribute__((device)) double __ocml_cos_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_cosh_f64(double);
__attribute__((device)) double __ocml_cospi_f64(double);
__attribute__((device)) double __ocml_i0_f64(double);
__attribute__((device)) double __ocml_i1_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_erfc_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_erfcinv_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_erfcx_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_erf_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_erfinv_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_exp10_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_exp2_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_exp_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_expm1_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_fabs_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_fdim_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_floor_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_fma_f64(double, double, double);
__attribute__((device)) __attribute__((const)) double __ocml_fmax_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_fmin_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_fmod_f64(double, double);
__attribute__((device)) double __ocml_frexp_f64(double,
                                   __attribute__((address_space(5))) int *);
__attribute__((device)) __attribute__((const)) double __ocml_hypot_f64(double, double);
__attribute__((device)) __attribute__((const)) int __ocml_ilogb_f64(double);
__attribute__((device)) __attribute__((const)) int __ocml_isfinite_f64(double);
__attribute__((device)) __attribute__((const)) int __ocml_isinf_f64(double);
__attribute__((device)) __attribute__((const)) int __ocml_isnan_f64(double);
__attribute__((device)) double __ocml_j0_f64(double);
__attribute__((device)) double __ocml_j1_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_ldexp_f64(double, int);
__attribute__((device)) double __ocml_lgamma_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_log10_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_log1p_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_log2_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_logb_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_log_f64(double);
__attribute__((device)) double __ocml_modf_f64(double,
                                  __attribute__((address_space(5))) double *);
__attribute__((device)) __attribute__((const)) double __ocml_nearbyint_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_nextafter_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_len3_f64(double, double,
                                                         double);
__attribute__((device)) __attribute__((const)) double __ocml_len4_f64(double, double, double,
                                                         double);
__attribute__((device)) __attribute__((pure)) double __ocml_ncdf_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_ncdfinv_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_pow_f64(double, double);
__attribute__((device)) __attribute__((pure)) double __ocml_pown_f64(double, int);
__attribute__((device)) __attribute__((pure)) double __ocml_rcbrt_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_remainder_f64(double, double);
__attribute__((device)) double __ocml_remquo_f64(double, double,
                                    __attribute__((address_space(5))) int *);
__attribute__((device)) __attribute__((const)) double __ocml_rhypot_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_rint_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_rlen3_f64(double, double,
                                                          double);
__attribute__((device)) __attribute__((const)) double __ocml_rlen4_f64(double, double,
                                                          double, double);
__attribute__((device)) __attribute__((const)) double __ocml_round_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_rsqrt_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_scalb_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_scalbn_f64(double, int);
__attribute__((device)) __attribute__((const)) int __ocml_signbit_f64(double);
__attribute__((device)) double __ocml_sincos_f64(double,
                                    __attribute__((address_space(5))) double *);
__attribute__((device)) double
__ocml_sincospi_f64(double, __attribute__((address_space(5))) double *);
__attribute__((device)) double __ocml_sin_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_sinh_f64(double);
__attribute__((device)) double __ocml_sinpi_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_sqrt_f64(double);
__attribute__((device)) double __ocml_tan_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_tanh_f64(double);
__attribute__((device)) double __ocml_tgamma_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_trunc_f64(double);
__attribute__((device)) double __ocml_y0_f64(double);
__attribute__((device)) double __ocml_y1_f64(double);


__attribute__((device)) __attribute__((const)) double __ocml_add_rte_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_add_rtn_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_add_rtp_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_add_rtz_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_sub_rte_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_sub_rtn_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_sub_rtp_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_sub_rtz_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_mul_rte_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_mul_rtn_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_mul_rtp_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_mul_rtz_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_div_rte_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_div_rtn_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_div_rtp_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_div_rtz_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_sqrt_rte_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_sqrt_rtn_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_sqrt_rtp_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_sqrt_rtz_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_fma_rte_f64(double, double,
                                                            double);
__attribute__((device)) __attribute__((const)) double __ocml_fma_rtn_f64(double, double,
                                                            double);
__attribute__((device)) __attribute__((const)) double __ocml_fma_rtp_f64(double, double,
                                                            double);
__attribute__((device)) __attribute__((const)) double __ocml_fma_rtz_f64(double, double,
                                                            double);

__attribute__((device)) __attribute__((const)) double
__llvm_amdgcn_rcp_f64(double) __asm("llvm.amdgcn.rcp.f64");
__attribute__((device)) __attribute__((const)) double
__llvm_amdgcn_rsq_f64(double) __asm("llvm.amdgcn.rsq.f64");

__attribute__((device)) __attribute__((const)) _Float16 __ocml_ceil_f16(_Float16);
__attribute__((device)) _Float16 __ocml_cos_f16(_Float16);
__attribute__((device)) __attribute__((pure)) _Float16 __ocml_exp_f16(_Float16);
__attribute__((device)) __attribute__((pure)) _Float16 __ocml_exp10_f16(_Float16);
__attribute__((device)) __attribute__((pure)) _Float16 __ocml_exp2_f16(_Float16);
__attribute__((device)) __attribute__((const)) _Float16 __ocml_floor_f16(_Float16);
__attribute__((device)) __attribute__((const)) _Float16 __ocml_fma_f16(_Float16, _Float16,
                                                          _Float16);
__attribute__((device)) __attribute__((const)) _Float16 __ocml_fabs_f16(_Float16);
__attribute__((device)) __attribute__((const)) int __ocml_isinf_f16(_Float16);
__attribute__((device)) __attribute__((const)) int __ocml_isnan_f16(_Float16);
__attribute__((device)) __attribute__((pure)) _Float16 __ocml_log_f16(_Float16);
__attribute__((device)) __attribute__((pure)) _Float16 __ocml_log10_f16(_Float16);
__attribute__((device)) __attribute__((pure)) _Float16 __ocml_log2_f16(_Float16);
__attribute__((device)) __attribute__((const)) _Float16 __llvm_amdgcn_rcp_f16(_Float16);
__attribute__((device)) __attribute__((const)) _Float16 __ocml_rint_f16(_Float16);
__attribute__((device)) __attribute__((const)) _Float16 __ocml_rsqrt_f16(_Float16);
__attribute__((device)) _Float16 __ocml_sin_f16(_Float16);
__attribute__((device)) __attribute__((const)) _Float16 __ocml_sqrt_f16(_Float16);
__attribute__((device)) __attribute__((const)) _Float16 __ocml_trunc_f16(_Float16);
__attribute__((device)) __attribute__((pure)) _Float16 __ocml_pown_f16(_Float16, int);

typedef _Float16 __2f16 __attribute__((ext_vector_type(2)));
typedef short __2i16 __attribute__((ext_vector_type(2)));





__attribute__((device)) __attribute__((const)) float __ockl_fdot2(__2f16 a, __2f16 b,
                                                     float c, unsigned int s);

__attribute__((device)) __attribute__((const)) __2f16 __ocml_ceil_2f16(__2f16);
__attribute__((device)) __attribute__((const)) __2f16 __ocml_fabs_2f16(__2f16);
__attribute__((device)) __2f16 __ocml_cos_2f16(__2f16);
__attribute__((device)) __attribute__((pure)) __2f16 __ocml_exp_2f16(__2f16);
__attribute__((device)) __attribute__((pure)) __2f16 __ocml_exp10_2f16(__2f16);
__attribute__((device)) __attribute__((pure)) __2f16 __ocml_exp2_2f16(__2f16);
__attribute__((device)) __attribute__((const)) __2f16 __ocml_floor_2f16(__2f16);
__attribute__((device)) __attribute__((const))
__2f16 __ocml_fma_2f16(__2f16, __2f16, __2f16);
__attribute__((device)) __attribute__((const)) __2i16 __ocml_isinf_2f16(__2f16);
__attribute__((device)) __attribute__((const)) __2i16 __ocml_isnan_2f16(__2f16);
__attribute__((device)) __attribute__((pure)) __2f16 __ocml_log_2f16(__2f16);
__attribute__((device)) __attribute__((pure)) __2f16 __ocml_log10_2f16(__2f16);
__attribute__((device)) __attribute__((pure)) __2f16 __ocml_log2_2f16(__2f16);
__attribute__((device)) inline __2f16
__llvm_amdgcn_rcp_2f16(__2f16 __x)
{
  return (__2f16)(__llvm_amdgcn_rcp_f16(__x.x), __llvm_amdgcn_rcp_f16(__x.y));
}
__attribute__((device)) __attribute__((const)) __2f16 __ocml_rint_2f16(__2f16);
__attribute__((device)) __attribute__((const)) __2f16 __ocml_rsqrt_2f16(__2f16);
__attribute__((device)) __2f16 __ocml_sin_2f16(__2f16);
__attribute__((device)) __attribute__((const)) __2f16 __ocml_sqrt_2f16(__2f16);
__attribute__((device)) __attribute__((const)) __2f16 __ocml_trunc_2f16(__2f16);
__attribute__((device)) __attribute__((const)) __2f16 __ocml_pown_2f16(__2f16, __2i16);
# 74 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/openmp_wrappers/__clang_openmp_device_functions.h" 2 3





#pragma omp end declare variant
# 2 "<built-in>" 2
# 1 "c_timers.c" 2
# 1 "../common/wtime.h" 1
# 2 "c_timers.c" 2
# 1 "/usr/include/stdlib.h" 1 3 4
# 25 "/usr/include/stdlib.h" 3 4
# 1 "/usr/include/bits/libc-header-start.h" 1 3 4
# 26 "/usr/include/stdlib.h" 2 3 4





# 1 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/stddef.h" 1 3 4
# 46 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/stddef.h" 3 4
typedef long unsigned int size_t;
# 74 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/stddef.h" 3 4
typedef int wchar_t;
# 32 "/usr/include/stdlib.h" 2 3 4







# 1 "/usr/include/bits/waitflags.h" 1 3 4
# 40 "/usr/include/stdlib.h" 2 3 4
# 1 "/usr/include/bits/waitstatus.h" 1 3 4
# 41 "/usr/include/stdlib.h" 2 3 4
# 55 "/usr/include/stdlib.h" 3 4
# 1 "/usr/include/bits/floatn.h" 1 3 4
# 119 "/usr/include/bits/floatn.h" 3 4
# 1 "/usr/include/bits/floatn-common.h" 1 3 4
# 24 "/usr/include/bits/floatn-common.h" 3 4
# 1 "/usr/include/bits/long-double.h" 1 3 4
# 25 "/usr/include/bits/floatn-common.h" 2 3 4
# 214 "/usr/include/bits/floatn-common.h" 3 4
typedef float _Float32;
# 251 "/usr/include/bits/floatn-common.h" 3 4
typedef double _Float64;
# 268 "/usr/include/bits/floatn-common.h" 3 4
typedef double _Float32x;
# 285 "/usr/include/bits/floatn-common.h" 3 4
typedef long double _Float64x;
# 120 "/usr/include/bits/floatn.h" 2 3 4
# 56 "/usr/include/stdlib.h" 2 3 4


typedef struct
  {
    int quot;
    int rem;
  } div_t;



typedef struct
  {
    long int quot;
    long int rem;
  } ldiv_t;





__extension__ typedef struct
  {
    long long int quot;
    long long int rem;
  } lldiv_t;
# 97 "/usr/include/stdlib.h" 3 4
extern size_t __ctype_get_mb_cur_max (void) __attribute__ ((__nothrow__ )) ;



extern double atof (const char *__nptr)
     __attribute__ ((__nothrow__ )) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) ;

extern int atoi (const char *__nptr)
     __attribute__ ((__nothrow__ )) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) ;

extern long int atol (const char *__nptr)
     __attribute__ ((__nothrow__ )) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) ;



__extension__ extern long long int atoll (const char *__nptr)
     __attribute__ ((__nothrow__ )) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) ;



extern double strtod (const char *__restrict __nptr,
        char **__restrict __endptr)
     __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1)));



extern float strtof (const char *__restrict __nptr,
       char **__restrict __endptr) __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1)));

extern long double strtold (const char *__restrict __nptr,
       char **__restrict __endptr)
     __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1)));
# 176 "/usr/include/stdlib.h" 3 4
extern long int strtol (const char *__restrict __nptr,
   char **__restrict __endptr, int __base)
     __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1)));

extern unsigned long int strtoul (const char *__restrict __nptr,
      char **__restrict __endptr, int __base)
     __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1)));



__extension__
extern long long int strtoq (const char *__restrict __nptr,
        char **__restrict __endptr, int __base)
     __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1)));

__extension__
extern unsigned long long int strtouq (const char *__restrict __nptr,
           char **__restrict __endptr, int __base)
     __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1)));




__extension__
extern long long int strtoll (const char *__restrict __nptr,
         char **__restrict __endptr, int __base)
     __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1)));

__extension__
extern unsigned long long int strtoull (const char *__restrict __nptr,
     char **__restrict __endptr, int __base)
     __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1)));
# 385 "/usr/include/stdlib.h" 3 4
extern char *l64a (long int __n) __attribute__ ((__nothrow__ )) ;


extern long int a64l (const char *__s)
     __attribute__ ((__nothrow__ )) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) ;





# 1 "/usr/include/sys/types.h" 1 3 4
# 33 "/usr/include/sys/types.h" 3 4
typedef __u_char u_char;
typedef __u_short u_short;
typedef __u_int u_int;
typedef __u_long u_long;
typedef __quad_t quad_t;
typedef __u_quad_t u_quad_t;
typedef __fsid_t fsid_t;


typedef __loff_t loff_t;




typedef __ino_t ino_t;
# 59 "/usr/include/sys/types.h" 3 4
typedef __dev_t dev_t;




typedef __gid_t gid_t;




typedef __mode_t mode_t;




typedef __nlink_t nlink_t;




typedef __uid_t uid_t;





typedef __off_t off_t;
# 97 "/usr/include/sys/types.h" 3 4
typedef __pid_t pid_t;





typedef __id_t id_t;




typedef __ssize_t ssize_t;





typedef __daddr_t daddr_t;
typedef __caddr_t caddr_t;





typedef __key_t key_t;





# 1 "/usr/include/bits/types/clock_t.h" 1 3 4






typedef __clock_t clock_t;
# 127 "/usr/include/sys/types.h" 2 3 4

# 1 "/usr/include/bits/types/clockid_t.h" 1 3 4






typedef __clockid_t clockid_t;
# 129 "/usr/include/sys/types.h" 2 3 4
# 1 "/usr/include/bits/types/time_t.h" 1 3 4






typedef __time_t time_t;
# 130 "/usr/include/sys/types.h" 2 3 4
# 1 "/usr/include/bits/types/timer_t.h" 1 3 4






typedef __timer_t timer_t;
# 131 "/usr/include/sys/types.h" 2 3 4
# 144 "/usr/include/sys/types.h" 3 4
# 1 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/stddef.h" 1 3 4
# 145 "/usr/include/sys/types.h" 2 3 4



typedef unsigned long int ulong;
typedef unsigned short int ushort;
typedef unsigned int uint;







typedef __uint8_t u_int8_t;
typedef __uint16_t u_int16_t;
typedef __uint32_t u_int32_t;
typedef __uint64_t u_int64_t;


typedef int register_t __attribute__ ((__mode__ (__word__)));
# 176 "/usr/include/sys/types.h" 3 4
# 1 "/usr/include/endian.h" 1 3 4
# 36 "/usr/include/endian.h" 3 4
# 1 "/usr/include/bits/endian.h" 1 3 4
# 37 "/usr/include/endian.h" 2 3 4
# 60 "/usr/include/endian.h" 3 4
# 1 "/usr/include/bits/byteswap.h" 1 3 4
# 33 "/usr/include/bits/byteswap.h" 3 4
static __inline __uint16_t
__bswap_16 (__uint16_t __bsx)
{



  return ((__uint16_t) ((((__bsx) >> 8) & 0xff) | (((__bsx) & 0xff) << 8)));

}






static __inline __uint32_t
__bswap_32 (__uint32_t __bsx)
{



  return ((((__bsx) & 0xff000000u) >> 24) | (((__bsx) & 0x00ff0000u) >> 8) | (((__bsx) & 0x0000ff00u) << 8) | (((__bsx) & 0x000000ffu) << 24));

}
# 69 "/usr/include/bits/byteswap.h" 3 4
__extension__ static __inline __uint64_t
__bswap_64 (__uint64_t __bsx)
{



  return ((((__bsx) & 0xff00000000000000ull) >> 56) | (((__bsx) & 0x00ff000000000000ull) >> 40) | (((__bsx) & 0x0000ff0000000000ull) >> 24) | (((__bsx) & 0x000000ff00000000ull) >> 8) | (((__bsx) & 0x00000000ff000000ull) << 8) | (((__bsx) & 0x0000000000ff0000ull) << 24) | (((__bsx) & 0x000000000000ff00ull) << 40) | (((__bsx) & 0x00000000000000ffull) << 56));

}
# 61 "/usr/include/endian.h" 2 3 4
# 1 "/usr/include/bits/uintn-identity.h" 1 3 4
# 32 "/usr/include/bits/uintn-identity.h" 3 4
static __inline __uint16_t
__uint16_identity (__uint16_t __x)
{
  return __x;
}

static __inline __uint32_t
__uint32_identity (__uint32_t __x)
{
  return __x;
}

static __inline __uint64_t
__uint64_identity (__uint64_t __x)
{
  return __x;
}
# 62 "/usr/include/endian.h" 2 3 4
# 177 "/usr/include/sys/types.h" 2 3 4


# 1 "/usr/include/sys/select.h" 1 3 4
# 30 "/usr/include/sys/select.h" 3 4
# 1 "/usr/include/bits/select.h" 1 3 4
# 22 "/usr/include/bits/select.h" 3 4
# 1 "/usr/include/bits/wordsize.h" 1 3 4
# 23 "/usr/include/bits/select.h" 2 3 4
# 31 "/usr/include/sys/select.h" 2 3 4


# 1 "/usr/include/bits/types/sigset_t.h" 1 3 4



# 1 "/usr/include/bits/types/__sigset_t.h" 1 3 4




typedef struct
{
  unsigned long int __val[(1024 / (8 * sizeof (unsigned long int)))];
} __sigset_t;
# 5 "/usr/include/bits/types/sigset_t.h" 2 3 4


typedef __sigset_t sigset_t;
# 34 "/usr/include/sys/select.h" 2 3 4



# 1 "/usr/include/bits/types/struct_timeval.h" 1 3 4







struct timeval
{
  __time_t tv_sec;
  __suseconds_t tv_usec;
};
# 38 "/usr/include/sys/select.h" 2 3 4

# 1 "/usr/include/bits/types/struct_timespec.h" 1 3 4








struct timespec
{
  __time_t tv_sec;
  __syscall_slong_t tv_nsec;
};
# 40 "/usr/include/sys/select.h" 2 3 4



typedef __suseconds_t suseconds_t;





typedef long int __fd_mask;
# 59 "/usr/include/sys/select.h" 3 4
typedef struct
  {






    __fd_mask __fds_bits[1024 / (8 * (int) sizeof (__fd_mask))];


  } fd_set;






typedef __fd_mask fd_mask;
# 101 "/usr/include/sys/select.h" 3 4
extern int select (int __nfds, fd_set *__restrict __readfds,
     fd_set *__restrict __writefds,
     fd_set *__restrict __exceptfds,
     struct timeval *__restrict __timeout);
# 113 "/usr/include/sys/select.h" 3 4
extern int pselect (int __nfds, fd_set *__restrict __readfds,
      fd_set *__restrict __writefds,
      fd_set *__restrict __exceptfds,
      const struct timespec *__restrict __timeout,
      const __sigset_t *__restrict __sigmask);
# 180 "/usr/include/sys/types.h" 2 3 4





typedef __blksize_t blksize_t;






typedef __blkcnt_t blkcnt_t;



typedef __fsblkcnt_t fsblkcnt_t;



typedef __fsfilcnt_t fsfilcnt_t;
# 227 "/usr/include/sys/types.h" 3 4
# 1 "/usr/include/bits/pthreadtypes.h" 1 3 4
# 23 "/usr/include/bits/pthreadtypes.h" 3 4
# 1 "/usr/include/bits/thread-shared-types.h" 1 3 4
# 77 "/usr/include/bits/thread-shared-types.h" 3 4
# 1 "/usr/include/bits/pthreadtypes-arch.h" 1 3 4
# 21 "/usr/include/bits/pthreadtypes-arch.h" 3 4
# 1 "/usr/include/bits/wordsize.h" 1 3 4
# 22 "/usr/include/bits/pthreadtypes-arch.h" 2 3 4
# 65 "/usr/include/bits/pthreadtypes-arch.h" 3 4
struct __pthread_rwlock_arch_t
{
  unsigned int __readers;
  unsigned int __writers;
  unsigned int __wrphase_futex;
  unsigned int __writers_futex;
  unsigned int __pad3;
  unsigned int __pad4;

  int __cur_writer;
  int __shared;
  signed char __rwelision;




  unsigned char __pad1[7];


  unsigned long int __pad2;


  unsigned int __flags;
# 99 "/usr/include/bits/pthreadtypes-arch.h" 3 4
};
# 78 "/usr/include/bits/thread-shared-types.h" 2 3 4




typedef struct __pthread_internal_list
{
  struct __pthread_internal_list *__prev;
  struct __pthread_internal_list *__next;
} __pthread_list_t;
# 118 "/usr/include/bits/thread-shared-types.h" 3 4
struct __pthread_mutex_s
{
  int __lock ;
  unsigned int __count;
  int __owner;

  unsigned int __nusers;
# 148 "/usr/include/bits/thread-shared-types.h" 3 4
  int __kind;





  short __spins; short __elision;
  __pthread_list_t __list;
# 166 "/usr/include/bits/thread-shared-types.h" 3 4
};




struct __pthread_cond_s
{
  __extension__ union
  {
    __extension__ unsigned long long int __wseq;
    struct
    {
      unsigned int __low;
      unsigned int __high;
    } __wseq32;
  };
  __extension__ union
  {
    __extension__ unsigned long long int __g1_start;
    struct
    {
      unsigned int __low;
      unsigned int __high;
    } __g1_start32;
  };
  unsigned int __g_refs[2] ;
  unsigned int __g_size[2];
  unsigned int __g1_orig_size;
  unsigned int __wrefs;
  unsigned int __g_signals[2];
};
# 24 "/usr/include/bits/pthreadtypes.h" 2 3 4



typedef unsigned long int pthread_t;




typedef union
{
  char __size[4];
  int __align;
} pthread_mutexattr_t;




typedef union
{
  char __size[4];
  int __align;
} pthread_condattr_t;



typedef unsigned int pthread_key_t;



typedef int pthread_once_t;


union pthread_attr_t
{
  char __size[56];
  long int __align;
};

typedef union pthread_attr_t pthread_attr_t;




typedef union
{
  struct __pthread_mutex_s __data;
  char __size[40];
  long int __align;
} pthread_mutex_t;


typedef union
{
  struct __pthread_cond_s __data;
  char __size[48];
  __extension__ long long int __align;
} pthread_cond_t;





typedef union
{
  struct __pthread_rwlock_arch_t __data;
  char __size[56];
  long int __align;
} pthread_rwlock_t;

typedef union
{
  char __size[8];
  long int __align;
} pthread_rwlockattr_t;





typedef volatile int pthread_spinlock_t;




typedef union
{
  char __size[32];
  long int __align;
} pthread_barrier_t;

typedef union
{
  char __size[4];
  int __align;
} pthread_barrierattr_t;
# 228 "/usr/include/sys/types.h" 2 3 4
# 395 "/usr/include/stdlib.h" 2 3 4






extern long int random (void) __attribute__ ((__nothrow__ ));


extern void srandom (unsigned int __seed) __attribute__ ((__nothrow__ ));





extern char *initstate (unsigned int __seed, char *__statebuf,
   size_t __statelen) __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (2)));



extern char *setstate (char *__statebuf) __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1)));







struct random_data
  {
    int32_t *fptr;
    int32_t *rptr;
    int32_t *state;
    int rand_type;
    int rand_deg;
    int rand_sep;
    int32_t *end_ptr;
  };

extern int random_r (struct random_data *__restrict __buf,
       int32_t *__restrict __result) __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1, 2)));

extern int srandom_r (unsigned int __seed, struct random_data *__buf)
     __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (2)));

extern int initstate_r (unsigned int __seed, char *__restrict __statebuf,
   size_t __statelen,
   struct random_data *__restrict __buf)
     __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (2, 4)));

extern int setstate_r (char *__restrict __statebuf,
         struct random_data *__restrict __buf)
     __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1, 2)));





extern int rand (void) __attribute__ ((__nothrow__ ));

extern void srand (unsigned int __seed) __attribute__ ((__nothrow__ ));



extern int rand_r (unsigned int *__seed) __attribute__ ((__nothrow__ ));







extern double drand48 (void) __attribute__ ((__nothrow__ ));
extern double erand48 (unsigned short int __xsubi[3]) __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1)));


extern long int lrand48 (void) __attribute__ ((__nothrow__ ));
extern long int nrand48 (unsigned short int __xsubi[3])
     __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1)));


extern long int mrand48 (void) __attribute__ ((__nothrow__ ));
extern long int jrand48 (unsigned short int __xsubi[3])
     __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1)));


extern void srand48 (long int __seedval) __attribute__ ((__nothrow__ ));
extern unsigned short int *seed48 (unsigned short int __seed16v[3])
     __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1)));
extern void lcong48 (unsigned short int __param[7]) __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1)));





struct drand48_data
  {
    unsigned short int __x[3];
    unsigned short int __old_x[3];
    unsigned short int __c;
    unsigned short int __init;
    __extension__ unsigned long long int __a;

  };


extern int drand48_r (struct drand48_data *__restrict __buffer,
        double *__restrict __result) __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1, 2)));
extern int erand48_r (unsigned short int __xsubi[3],
        struct drand48_data *__restrict __buffer,
        double *__restrict __result) __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1, 2)));


extern int lrand48_r (struct drand48_data *__restrict __buffer,
        long int *__restrict __result)
     __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1, 2)));
extern int nrand48_r (unsigned short int __xsubi[3],
        struct drand48_data *__restrict __buffer,
        long int *__restrict __result)
     __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1, 2)));


extern int mrand48_r (struct drand48_data *__restrict __buffer,
        long int *__restrict __result)
     __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1, 2)));
extern int jrand48_r (unsigned short int __xsubi[3],
        struct drand48_data *__restrict __buffer,
        long int *__restrict __result)
     __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1, 2)));


extern int srand48_r (long int __seedval, struct drand48_data *__buffer)
     __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (2)));

extern int seed48_r (unsigned short int __seed16v[3],
       struct drand48_data *__buffer) __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1, 2)));

extern int lcong48_r (unsigned short int __param[7],
        struct drand48_data *__buffer)
     __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1, 2)));




extern void *malloc (size_t __size) __attribute__ ((__nothrow__ )) __attribute__ ((__malloc__)) ;

extern void *calloc (size_t __nmemb, size_t __size)
     __attribute__ ((__nothrow__ )) __attribute__ ((__malloc__)) ;






extern void *realloc (void *__ptr, size_t __size)
     __attribute__ ((__nothrow__ )) __attribute__ ((__warn_unused_result__));
# 563 "/usr/include/stdlib.h" 3 4
extern void free (void *__ptr) __attribute__ ((__nothrow__ ));



# 1 "/usr/include/alloca.h" 1 3 4
# 24 "/usr/include/alloca.h" 3 4
# 1 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/stddef.h" 1 3 4
# 25 "/usr/include/alloca.h" 2 3 4







extern void *alloca (size_t __size) __attribute__ ((__nothrow__ ));
# 567 "/usr/include/stdlib.h" 2 3 4





extern void *valloc (size_t __size) __attribute__ ((__nothrow__ )) __attribute__ ((__malloc__)) ;




extern int posix_memalign (void **__memptr, size_t __alignment, size_t __size)
     __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1))) ;




extern void *aligned_alloc (size_t __alignment, size_t __size)
     __attribute__ ((__nothrow__ )) __attribute__ ((__malloc__)) ;



extern void abort (void) __attribute__ ((__nothrow__ )) __attribute__ ((__noreturn__));



extern int atexit (void (*__func) (void)) __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1)));







extern int at_quick_exit (void (*__func) (void)) __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1)));






extern int on_exit (void (*__func) (int __status, void *__arg), void *__arg)
     __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1)));





extern void exit (int __status) __attribute__ ((__nothrow__ )) __attribute__ ((__noreturn__));





extern void quick_exit (int __status) __attribute__ ((__nothrow__ )) __attribute__ ((__noreturn__));





extern void _Exit (int __status) __attribute__ ((__nothrow__ )) __attribute__ ((__noreturn__));




extern char *getenv (const char *__name) __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1))) ;
# 644 "/usr/include/stdlib.h" 3 4
extern int putenv (char *__string) __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1)));





extern int setenv (const char *__name, const char *__value, int __replace)
     __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (2)));


extern int unsetenv (const char *__name) __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1)));






extern int clearenv (void) __attribute__ ((__nothrow__ ));
# 672 "/usr/include/stdlib.h" 3 4
extern char *mktemp (char *__template) __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1)));
# 685 "/usr/include/stdlib.h" 3 4
extern int mkstemp (char *__template) __attribute__ ((__nonnull__ (1))) ;
# 707 "/usr/include/stdlib.h" 3 4
extern int mkstemps (char *__template, int __suffixlen) __attribute__ ((__nonnull__ (1))) ;
# 728 "/usr/include/stdlib.h" 3 4
extern char *mkdtemp (char *__template) __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1))) ;
# 781 "/usr/include/stdlib.h" 3 4
extern int system (const char *__command) ;
# 797 "/usr/include/stdlib.h" 3 4
extern char *realpath (const char *__restrict __name,
         char *__restrict __resolved) __attribute__ ((__nothrow__ )) ;






typedef int (*__compar_fn_t) (const void *, const void *);
# 817 "/usr/include/stdlib.h" 3 4
extern void *bsearch (const void *__key, const void *__base,
        size_t __nmemb, size_t __size, __compar_fn_t __compar)
     __attribute__ ((__nonnull__ (1, 2, 5))) ;







extern void qsort (void *__base, size_t __nmemb, size_t __size,
     __compar_fn_t __compar) __attribute__ ((__nonnull__ (1, 4)));
# 837 "/usr/include/stdlib.h" 3 4
extern int abs (int __x) __attribute__ ((__nothrow__ )) __attribute__ ((__const__)) ;
extern long int labs (long int __x) __attribute__ ((__nothrow__ )) __attribute__ ((__const__)) ;


__extension__ extern long long int llabs (long long int __x)
     __attribute__ ((__nothrow__ )) __attribute__ ((__const__)) ;






extern div_t div (int __numer, int __denom)
     __attribute__ ((__nothrow__ )) __attribute__ ((__const__)) ;
extern ldiv_t ldiv (long int __numer, long int __denom)
     __attribute__ ((__nothrow__ )) __attribute__ ((__const__)) ;


__extension__ extern lldiv_t lldiv (long long int __numer,
        long long int __denom)
     __attribute__ ((__nothrow__ )) __attribute__ ((__const__)) ;
# 869 "/usr/include/stdlib.h" 3 4
extern char *ecvt (double __value, int __ndigit, int *__restrict __decpt,
     int *__restrict __sign) __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (3, 4))) ;




extern char *fcvt (double __value, int __ndigit, int *__restrict __decpt,
     int *__restrict __sign) __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (3, 4))) ;




extern char *gcvt (double __value, int __ndigit, char *__buf)
     __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (3))) ;




extern char *qecvt (long double __value, int __ndigit,
      int *__restrict __decpt, int *__restrict __sign)
     __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (3, 4))) ;
extern char *qfcvt (long double __value, int __ndigit,
      int *__restrict __decpt, int *__restrict __sign)
     __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (3, 4))) ;
extern char *qgcvt (long double __value, int __ndigit, char *__buf)
     __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (3))) ;




extern int ecvt_r (double __value, int __ndigit, int *__restrict __decpt,
     int *__restrict __sign, char *__restrict __buf,
     size_t __len) __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (3, 4, 5)));
extern int fcvt_r (double __value, int __ndigit, int *__restrict __decpt,
     int *__restrict __sign, char *__restrict __buf,
     size_t __len) __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (3, 4, 5)));

extern int qecvt_r (long double __value, int __ndigit,
      int *__restrict __decpt, int *__restrict __sign,
      char *__restrict __buf, size_t __len)
     __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (3, 4, 5)));
extern int qfcvt_r (long double __value, int __ndigit,
      int *__restrict __decpt, int *__restrict __sign,
      char *__restrict __buf, size_t __len)
     __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (3, 4, 5)));





extern int mblen (const char *__s, size_t __n) __attribute__ ((__nothrow__ ));


extern int mbtowc (wchar_t *__restrict __pwc,
     const char *__restrict __s, size_t __n) __attribute__ ((__nothrow__ ));


extern int wctomb (char *__s, wchar_t __wchar) __attribute__ ((__nothrow__ ));



extern size_t mbstowcs (wchar_t *__restrict __pwcs,
   const char *__restrict __s, size_t __n) __attribute__ ((__nothrow__ ));

extern size_t wcstombs (char *__restrict __s,
   const wchar_t *__restrict __pwcs, size_t __n)
     __attribute__ ((__nothrow__ ));







extern int rpmatch (const char *__response) __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1))) ;
# 954 "/usr/include/stdlib.h" 3 4
extern int getsubopt (char **__restrict __optionp,
        char *const *__restrict __tokens,
        char **__restrict __valuep)
     __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1, 2, 3))) ;
# 1000 "/usr/include/stdlib.h" 3 4
extern int getloadavg (double __loadavg[], int __nelem)
     __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1)));
# 1010 "/usr/include/stdlib.h" 3 4
# 1 "/usr/include/bits/stdlib-float.h" 1 3 4
# 1011 "/usr/include/stdlib.h" 2 3 4
# 3 "c_timers.c" 2


void wtime_( double * );





static double elapsed_time( void )
{
    double t;

    wtime_( &t );
    return( t );
}


static double start[64], elapsed[64];




void timer_clear( int n )
{
    elapsed[n] = 0.0;
}





void timer_start( int n )
{
    start[n] = elapsed_time();
}





void timer_stop( int n )
{
    double t, now;

    now = elapsed_time();
    t = now - start[n];
    elapsed[n] += t;

}





double timer_read( int n )
{
    return( elapsed[n] );
}
