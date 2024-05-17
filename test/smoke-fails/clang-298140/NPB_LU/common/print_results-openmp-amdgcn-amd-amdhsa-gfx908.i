# 1 "print_results.c"
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
# 1 "print_results.c" 2
# 1 "/usr/include/stdio.h" 1 3 4
# 27 "/usr/include/stdio.h" 3 4
# 1 "/usr/include/bits/libc-header-start.h" 1 3 4
# 28 "/usr/include/stdio.h" 2 3 4





# 1 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/stddef.h" 1 3 4
# 46 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/stddef.h" 3 4
typedef long unsigned int size_t;
# 34 "/usr/include/stdio.h" 2 3 4


# 1 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/stdarg.h" 1 3 4
# 14 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/stdarg.h" 3 4
typedef __builtin_va_list va_list;
# 32 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/stdarg.h" 3 4
typedef __builtin_va_list __gnuc_va_list;
# 37 "/usr/include/stdio.h" 2 3 4


# 1 "/usr/include/bits/types/__fpos_t.h" 1 3 4




# 1 "/usr/include/bits/types/__mbstate_t.h" 1 3 4
# 13 "/usr/include/bits/types/__mbstate_t.h" 3 4
typedef struct
{
  int __count;
  union
  {
    unsigned int __wch;
    char __wchb[4];
  } __value;
} __mbstate_t;
# 6 "/usr/include/bits/types/__fpos_t.h" 2 3 4




typedef struct _G_fpos_t
{
  __off_t __pos;
  __mbstate_t __state;
} __fpos_t;
# 40 "/usr/include/stdio.h" 2 3 4
# 1 "/usr/include/bits/types/__fpos64_t.h" 1 3 4
# 10 "/usr/include/bits/types/__fpos64_t.h" 3 4
typedef struct _G_fpos64_t
{
  __off64_t __pos;
  __mbstate_t __state;
} __fpos64_t;
# 41 "/usr/include/stdio.h" 2 3 4
# 1 "/usr/include/bits/types/__FILE.h" 1 3 4



struct _IO_FILE;
typedef struct _IO_FILE __FILE;
# 42 "/usr/include/stdio.h" 2 3 4
# 1 "/usr/include/bits/types/FILE.h" 1 3 4



struct _IO_FILE;


typedef struct _IO_FILE FILE;
# 43 "/usr/include/stdio.h" 2 3 4
# 1 "/usr/include/bits/types/struct_FILE.h" 1 3 4
# 35 "/usr/include/bits/types/struct_FILE.h" 3 4
struct _IO_FILE;
struct _IO_marker;
struct _IO_codecvt;
struct _IO_wide_data;




typedef void _IO_lock_t;





struct _IO_FILE
{
  int _flags;


  char *_IO_read_ptr;
  char *_IO_read_end;
  char *_IO_read_base;
  char *_IO_write_base;
  char *_IO_write_ptr;
  char *_IO_write_end;
  char *_IO_buf_base;
  char *_IO_buf_end;


  char *_IO_save_base;
  char *_IO_backup_base;
  char *_IO_save_end;

  struct _IO_marker *_markers;

  struct _IO_FILE *_chain;

  int _fileno;
  int _flags2;
  __off_t _old_offset;


  unsigned short _cur_column;
  signed char _vtable_offset;
  char _shortbuf[1];

  _IO_lock_t *_lock;







  __off64_t _offset;

  struct _IO_codecvt *_codecvt;
  struct _IO_wide_data *_wide_data;
  struct _IO_FILE *_freeres_list;
  void *_freeres_buf;
  size_t __pad5;
  int _mode;

  char _unused2[15 * sizeof (int) - 4 * sizeof (void *) - sizeof (size_t)];
};
# 44 "/usr/include/stdio.h" 2 3 4








typedef __gnuc_va_list va_list;
# 63 "/usr/include/stdio.h" 3 4
typedef __off_t off_t;
# 77 "/usr/include/stdio.h" 3 4
typedef __ssize_t ssize_t;






typedef __fpos_t fpos_t;
# 133 "/usr/include/stdio.h" 3 4
# 1 "/usr/include/bits/stdio_lim.h" 1 3 4
# 134 "/usr/include/stdio.h" 2 3 4



extern FILE *stdin;
extern FILE *stdout;
extern FILE *stderr;






extern int remove (const char *__filename) __attribute__ ((__nothrow__ ));

extern int rename (const char *__old, const char *__new) __attribute__ ((__nothrow__ ));



extern int renameat (int __oldfd, const char *__old, int __newfd,
       const char *__new) __attribute__ ((__nothrow__ ));
# 173 "/usr/include/stdio.h" 3 4
extern FILE *tmpfile (void) ;
# 187 "/usr/include/stdio.h" 3 4
extern char *tmpnam (char *__s) __attribute__ ((__nothrow__ )) ;




extern char *tmpnam_r (char *__s) __attribute__ ((__nothrow__ )) ;
# 204 "/usr/include/stdio.h" 3 4
extern char *tempnam (const char *__dir, const char *__pfx)
     __attribute__ ((__nothrow__ )) __attribute__ ((__malloc__)) ;







extern int fclose (FILE *__stream);




extern int fflush (FILE *__stream);
# 227 "/usr/include/stdio.h" 3 4
extern int fflush_unlocked (FILE *__stream);
# 246 "/usr/include/stdio.h" 3 4
extern FILE *fopen (const char *__restrict __filename,
      const char *__restrict __modes) ;




extern FILE *freopen (const char *__restrict __filename,
        const char *__restrict __modes,
        FILE *__restrict __stream) ;
# 279 "/usr/include/stdio.h" 3 4
extern FILE *fdopen (int __fd, const char *__modes) __attribute__ ((__nothrow__ )) ;
# 292 "/usr/include/stdio.h" 3 4
extern FILE *fmemopen (void *__s, size_t __len, const char *__modes)
  __attribute__ ((__nothrow__ )) ;




extern FILE *open_memstream (char **__bufloc, size_t *__sizeloc) __attribute__ ((__nothrow__ )) ;





extern void setbuf (FILE *__restrict __stream, char *__restrict __buf) __attribute__ ((__nothrow__ ));



extern int setvbuf (FILE *__restrict __stream, char *__restrict __buf,
      int __modes, size_t __n) __attribute__ ((__nothrow__ ));




extern void setbuffer (FILE *__restrict __stream, char *__restrict __buf,
         size_t __size) __attribute__ ((__nothrow__ ));


extern void setlinebuf (FILE *__stream) __attribute__ ((__nothrow__ ));







extern int fprintf (FILE *__restrict __stream,
      const char *__restrict __format, ...);




extern int printf (const char *__restrict __format, ...);

extern int sprintf (char *__restrict __s,
      const char *__restrict __format, ...) __attribute__ ((__nothrow__));





extern int vfprintf (FILE *__restrict __s, const char *__restrict __format,
       __gnuc_va_list __arg);




extern int vprintf (const char *__restrict __format, __gnuc_va_list __arg);

extern int vsprintf (char *__restrict __s, const char *__restrict __format,
       __gnuc_va_list __arg) __attribute__ ((__nothrow__));



extern int snprintf (char *__restrict __s, size_t __maxlen,
       const char *__restrict __format, ...)
     __attribute__ ((__nothrow__)) __attribute__ ((__format__ (__printf__, 3, 4)));

extern int vsnprintf (char *__restrict __s, size_t __maxlen,
        const char *__restrict __format, __gnuc_va_list __arg)
     __attribute__ ((__nothrow__)) __attribute__ ((__format__ (__printf__, 3, 0)));
# 379 "/usr/include/stdio.h" 3 4
extern int vdprintf (int __fd, const char *__restrict __fmt,
       __gnuc_va_list __arg)
     __attribute__ ((__format__ (__printf__, 2, 0)));
extern int dprintf (int __fd, const char *__restrict __fmt, ...)
     __attribute__ ((__format__ (__printf__, 2, 3)));







extern int fscanf (FILE *__restrict __stream,
     const char *__restrict __format, ...) ;




extern int scanf (const char *__restrict __format, ...) ;

extern int sscanf (const char *__restrict __s,
     const char *__restrict __format, ...) __attribute__ ((__nothrow__ ));
# 409 "/usr/include/stdio.h" 3 4
extern int fscanf (FILE *__restrict __stream, const char *__restrict __format, ...) __asm__ ("" "__isoc99_fscanf") ;


extern int scanf (const char *__restrict __format, ...) __asm__ ("" "__isoc99_scanf") ;

extern int sscanf (const char *__restrict __s, const char *__restrict __format, ...) __asm__ ("" "__isoc99_sscanf") __attribute__ ((__nothrow__ ));
# 434 "/usr/include/stdio.h" 3 4
extern int vfscanf (FILE *__restrict __s, const char *__restrict __format,
      __gnuc_va_list __arg)
     __attribute__ ((__format__ (__scanf__, 2, 0))) ;





extern int vscanf (const char *__restrict __format, __gnuc_va_list __arg)
     __attribute__ ((__format__ (__scanf__, 1, 0))) ;


extern int vsscanf (const char *__restrict __s,
      const char *__restrict __format, __gnuc_va_list __arg)
     __attribute__ ((__nothrow__ )) __attribute__ ((__format__ (__scanf__, 2, 0)));
# 457 "/usr/include/stdio.h" 3 4
extern int vfscanf (FILE *__restrict __s, const char *__restrict __format, __gnuc_va_list __arg) __asm__ ("" "__isoc99_vfscanf")



     __attribute__ ((__format__ (__scanf__, 2, 0))) ;
extern int vscanf (const char *__restrict __format, __gnuc_va_list __arg) __asm__ ("" "__isoc99_vscanf")

     __attribute__ ((__format__ (__scanf__, 1, 0))) ;
extern int vsscanf (const char *__restrict __s, const char *__restrict __format, __gnuc_va_list __arg) __asm__ ("" "__isoc99_vsscanf") __attribute__ ((__nothrow__ ))



     __attribute__ ((__format__ (__scanf__, 2, 0)));
# 491 "/usr/include/stdio.h" 3 4
extern int fgetc (FILE *__stream);
extern int getc (FILE *__stream);





extern int getchar (void);






extern int getc_unlocked (FILE *__stream);
extern int getchar_unlocked (void);
# 516 "/usr/include/stdio.h" 3 4
extern int fgetc_unlocked (FILE *__stream);
# 527 "/usr/include/stdio.h" 3 4
extern int fputc (int __c, FILE *__stream);
extern int putc (int __c, FILE *__stream);





extern int putchar (int __c);
# 543 "/usr/include/stdio.h" 3 4
extern int fputc_unlocked (int __c, FILE *__stream);







extern int putc_unlocked (int __c, FILE *__stream);
extern int putchar_unlocked (int __c);






extern int getw (FILE *__stream);


extern int putw (int __w, FILE *__stream);







extern char *fgets (char *__restrict __s, int __n, FILE *__restrict __stream)
          ;
# 609 "/usr/include/stdio.h" 3 4
extern __ssize_t __getdelim (char **__restrict __lineptr,
                             size_t *__restrict __n, int __delimiter,
                             FILE *__restrict __stream) ;
extern __ssize_t getdelim (char **__restrict __lineptr,
                           size_t *__restrict __n, int __delimiter,
                           FILE *__restrict __stream) ;







extern __ssize_t getline (char **__restrict __lineptr,
                          size_t *__restrict __n,
                          FILE *__restrict __stream) ;







extern int fputs (const char *__restrict __s, FILE *__restrict __stream);





extern int puts (const char *__s);






extern int ungetc (int __c, FILE *__stream);






extern size_t fread (void *__restrict __ptr, size_t __size,
       size_t __n, FILE *__restrict __stream) ;




extern size_t fwrite (const void *__restrict __ptr, size_t __size,
        size_t __n, FILE *__restrict __s);
# 679 "/usr/include/stdio.h" 3 4
extern size_t fread_unlocked (void *__restrict __ptr, size_t __size,
         size_t __n, FILE *__restrict __stream) ;
extern size_t fwrite_unlocked (const void *__restrict __ptr, size_t __size,
          size_t __n, FILE *__restrict __stream);







extern int fseek (FILE *__stream, long int __off, int __whence);




extern long int ftell (FILE *__stream) ;




extern void rewind (FILE *__stream);
# 713 "/usr/include/stdio.h" 3 4
extern int fseeko (FILE *__stream, __off_t __off, int __whence);




extern __off_t ftello (FILE *__stream) ;
# 737 "/usr/include/stdio.h" 3 4
extern int fgetpos (FILE *__restrict __stream, fpos_t *__restrict __pos);




extern int fsetpos (FILE *__stream, const fpos_t *__pos);
# 763 "/usr/include/stdio.h" 3 4
extern void clearerr (FILE *__stream) __attribute__ ((__nothrow__ ));

extern int feof (FILE *__stream) __attribute__ ((__nothrow__ )) ;

extern int ferror (FILE *__stream) __attribute__ ((__nothrow__ )) ;



extern void clearerr_unlocked (FILE *__stream) __attribute__ ((__nothrow__ ));
extern int feof_unlocked (FILE *__stream) __attribute__ ((__nothrow__ )) ;
extern int ferror_unlocked (FILE *__stream) __attribute__ ((__nothrow__ )) ;







extern void perror (const char *__s);






# 1 "/usr/include/bits/sys_errlist.h" 1 3 4
# 26 "/usr/include/bits/sys_errlist.h" 3 4
extern int sys_nerr;
extern const char *const sys_errlist[];
# 788 "/usr/include/stdio.h" 2 3 4




extern int fileno (FILE *__stream) __attribute__ ((__nothrow__ )) ;




extern int fileno_unlocked (FILE *__stream) __attribute__ ((__nothrow__ )) ;
# 806 "/usr/include/stdio.h" 3 4
extern FILE *popen (const char *__command, const char *__modes) ;





extern int pclose (FILE *__stream);





extern char *ctermid (char *__s) __attribute__ ((__nothrow__ ));
# 846 "/usr/include/stdio.h" 3 4
extern void flockfile (FILE *__stream) __attribute__ ((__nothrow__ ));



extern int ftrylockfile (FILE *__stream) __attribute__ ((__nothrow__ )) ;


extern void funlockfile (FILE *__stream) __attribute__ ((__nothrow__ ));
# 864 "/usr/include/stdio.h" 3 4
extern int __uflow (FILE *);
extern int __overflow (FILE *, int);
# 2 "print_results.c" 2
# 1 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/openmp_wrappers/math.h" 1 3
# 34 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/openmp_wrappers/math.h" 3
# 1 "/usr/include/math.h" 1 3 4
# 27 "/usr/include/math.h" 3 4
# 1 "/usr/include/bits/libc-header-start.h" 1 3 4
# 28 "/usr/include/math.h" 2 3 4
# 40 "/usr/include/math.h" 3 4
# 1 "/usr/include/bits/math-vector.h" 1 3 4
# 25 "/usr/include/bits/math-vector.h" 3 4
# 1 "/usr/include/bits/libm-simd-decl-stubs.h" 1 3 4
# 26 "/usr/include/bits/math-vector.h" 2 3 4
# 41 "/usr/include/math.h" 2 3 4


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
# 44 "/usr/include/math.h" 2 3 4
# 138 "/usr/include/math.h" 3 4
# 1 "/usr/include/bits/flt-eval-method.h" 1 3 4
# 139 "/usr/include/math.h" 2 3 4
# 149 "/usr/include/math.h" 3 4
typedef float float_t;
typedef double double_t;
# 190 "/usr/include/math.h" 3 4
# 1 "/usr/include/bits/fp-logb.h" 1 3 4
# 191 "/usr/include/math.h" 2 3 4
# 233 "/usr/include/math.h" 3 4
# 1 "/usr/include/bits/fp-fast.h" 1 3 4
# 234 "/usr/include/math.h" 2 3 4
# 289 "/usr/include/math.h" 3 4
# 1 "/usr/include/bits/mathcalls-helper-functions.h" 1 3 4
# 21 "/usr/include/bits/mathcalls-helper-functions.h" 3 4
extern int __fpclassify (double __value) __attribute__ ((__nothrow__ ))
     __attribute__ ((__const__));


extern int __signbit (double __value) __attribute__ ((__nothrow__ ))
     __attribute__ ((__const__));



extern int __isinf (double __value) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));


extern int __finite (double __value) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));


extern int __isnan (double __value) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));


extern int __iseqsig (double __x, double __y) __attribute__ ((__nothrow__ ));


extern int __issignaling (double __value) __attribute__ ((__nothrow__ ))
     __attribute__ ((__const__));
# 290 "/usr/include/math.h" 2 3 4
# 1 "/usr/include/bits/mathcalls.h" 1 3 4
# 53 "/usr/include/bits/mathcalls.h" 3 4
extern double acos (double __x) __attribute__ ((__nothrow__ )); extern double __acos (double __x) __attribute__ ((__nothrow__ ));

extern double asin (double __x) __attribute__ ((__nothrow__ )); extern double __asin (double __x) __attribute__ ((__nothrow__ ));

extern double atan (double __x) __attribute__ ((__nothrow__ )); extern double __atan (double __x) __attribute__ ((__nothrow__ ));

extern double atan2 (double __y, double __x) __attribute__ ((__nothrow__ )); extern double __atan2 (double __y, double __x) __attribute__ ((__nothrow__ ));


 extern double cos (double __x) __attribute__ ((__nothrow__ )); extern double __cos (double __x) __attribute__ ((__nothrow__ ));

 extern double sin (double __x) __attribute__ ((__nothrow__ )); extern double __sin (double __x) __attribute__ ((__nothrow__ ));

extern double tan (double __x) __attribute__ ((__nothrow__ )); extern double __tan (double __x) __attribute__ ((__nothrow__ ));




extern double cosh (double __x) __attribute__ ((__nothrow__ )); extern double __cosh (double __x) __attribute__ ((__nothrow__ ));

extern double sinh (double __x) __attribute__ ((__nothrow__ )); extern double __sinh (double __x) __attribute__ ((__nothrow__ ));

extern double tanh (double __x) __attribute__ ((__nothrow__ )); extern double __tanh (double __x) __attribute__ ((__nothrow__ ));
# 85 "/usr/include/bits/mathcalls.h" 3 4
extern double acosh (double __x) __attribute__ ((__nothrow__ )); extern double __acosh (double __x) __attribute__ ((__nothrow__ ));

extern double asinh (double __x) __attribute__ ((__nothrow__ )); extern double __asinh (double __x) __attribute__ ((__nothrow__ ));

extern double atanh (double __x) __attribute__ ((__nothrow__ )); extern double __atanh (double __x) __attribute__ ((__nothrow__ ));





 extern double exp (double __x) __attribute__ ((__nothrow__ )); extern double __exp (double __x) __attribute__ ((__nothrow__ ));


extern double frexp (double __x, int *__exponent) __attribute__ ((__nothrow__ )); extern double __frexp (double __x, int *__exponent) __attribute__ ((__nothrow__ ));


extern double ldexp (double __x, int __exponent) __attribute__ ((__nothrow__ )); extern double __ldexp (double __x, int __exponent) __attribute__ ((__nothrow__ ));


 extern double log (double __x) __attribute__ ((__nothrow__ )); extern double __log (double __x) __attribute__ ((__nothrow__ ));


extern double log10 (double __x) __attribute__ ((__nothrow__ )); extern double __log10 (double __x) __attribute__ ((__nothrow__ ));


extern double modf (double __x, double *__iptr) __attribute__ ((__nothrow__ )); extern double __modf (double __x, double *__iptr) __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (2)));
# 119 "/usr/include/bits/mathcalls.h" 3 4
extern double expm1 (double __x) __attribute__ ((__nothrow__ )); extern double __expm1 (double __x) __attribute__ ((__nothrow__ ));


extern double log1p (double __x) __attribute__ ((__nothrow__ )); extern double __log1p (double __x) __attribute__ ((__nothrow__ ));


extern double logb (double __x) __attribute__ ((__nothrow__ )); extern double __logb (double __x) __attribute__ ((__nothrow__ ));




extern double exp2 (double __x) __attribute__ ((__nothrow__ )); extern double __exp2 (double __x) __attribute__ ((__nothrow__ ));


extern double log2 (double __x) __attribute__ ((__nothrow__ )); extern double __log2 (double __x) __attribute__ ((__nothrow__ ));






 extern double pow (double __x, double __y) __attribute__ ((__nothrow__ )); extern double __pow (double __x, double __y) __attribute__ ((__nothrow__ ));


extern double sqrt (double __x) __attribute__ ((__nothrow__ )); extern double __sqrt (double __x) __attribute__ ((__nothrow__ ));



extern double hypot (double __x, double __y) __attribute__ ((__nothrow__ )); extern double __hypot (double __x, double __y) __attribute__ ((__nothrow__ ));




extern double cbrt (double __x) __attribute__ ((__nothrow__ )); extern double __cbrt (double __x) __attribute__ ((__nothrow__ ));






extern double ceil (double __x) __attribute__ ((__nothrow__ )) __attribute__ ((__const__)); extern double __ceil (double __x) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));


extern double fabs (double __x) __attribute__ ((__nothrow__ )) __attribute__ ((__const__)); extern double __fabs (double __x) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));


extern double floor (double __x) __attribute__ ((__nothrow__ )) __attribute__ ((__const__)); extern double __floor (double __x) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));


extern double fmod (double __x, double __y) __attribute__ ((__nothrow__ )); extern double __fmod (double __x, double __y) __attribute__ ((__nothrow__ ));
# 177 "/usr/include/bits/mathcalls.h" 3 4
extern int isinf (double __value) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));




extern int finite (double __value) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));


extern double drem (double __x, double __y) __attribute__ ((__nothrow__ )); extern double __drem (double __x, double __y) __attribute__ ((__nothrow__ ));



extern double significand (double __x) __attribute__ ((__nothrow__ )); extern double __significand (double __x) __attribute__ ((__nothrow__ ));






extern double copysign (double __x, double __y) __attribute__ ((__nothrow__ )) __attribute__ ((__const__)); extern double __copysign (double __x, double __y) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));




extern double nan (const char *__tagb) __attribute__ ((__nothrow__ )); extern double __nan (const char *__tagb) __attribute__ ((__nothrow__ ));
# 211 "/usr/include/bits/mathcalls.h" 3 4
extern int isnan (double __value) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));





extern double j0 (double) __attribute__ ((__nothrow__ )); extern double __j0 (double) __attribute__ ((__nothrow__ ));
extern double j1 (double) __attribute__ ((__nothrow__ )); extern double __j1 (double) __attribute__ ((__nothrow__ ));
extern double jn (int, double) __attribute__ ((__nothrow__ )); extern double __jn (int, double) __attribute__ ((__nothrow__ ));
extern double y0 (double) __attribute__ ((__nothrow__ )); extern double __y0 (double) __attribute__ ((__nothrow__ ));
extern double y1 (double) __attribute__ ((__nothrow__ )); extern double __y1 (double) __attribute__ ((__nothrow__ ));
extern double yn (int, double) __attribute__ ((__nothrow__ )); extern double __yn (int, double) __attribute__ ((__nothrow__ ));





extern double erf (double) __attribute__ ((__nothrow__ )); extern double __erf (double) __attribute__ ((__nothrow__ ));
extern double erfc (double) __attribute__ ((__nothrow__ )); extern double __erfc (double) __attribute__ ((__nothrow__ ));
extern double lgamma (double) __attribute__ ((__nothrow__ )); extern double __lgamma (double) __attribute__ ((__nothrow__ ));




extern double tgamma (double) __attribute__ ((__nothrow__ )); extern double __tgamma (double) __attribute__ ((__nothrow__ ));





extern double gamma (double) __attribute__ ((__nothrow__ )); extern double __gamma (double) __attribute__ ((__nothrow__ ));







extern double lgamma_r (double, int *__signgamp) __attribute__ ((__nothrow__ )); extern double __lgamma_r (double, int *__signgamp) __attribute__ ((__nothrow__ ));






extern double rint (double __x) __attribute__ ((__nothrow__ )); extern double __rint (double __x) __attribute__ ((__nothrow__ ));


extern double nextafter (double __x, double __y) __attribute__ ((__nothrow__ )); extern double __nextafter (double __x, double __y) __attribute__ ((__nothrow__ ));

extern double nexttoward (double __x, long double __y) __attribute__ ((__nothrow__ )); extern double __nexttoward (double __x, long double __y) __attribute__ ((__nothrow__ ));
# 272 "/usr/include/bits/mathcalls.h" 3 4
extern double remainder (double __x, double __y) __attribute__ ((__nothrow__ )); extern double __remainder (double __x, double __y) __attribute__ ((__nothrow__ ));



extern double scalbn (double __x, int __n) __attribute__ ((__nothrow__ )); extern double __scalbn (double __x, int __n) __attribute__ ((__nothrow__ ));



extern int ilogb (double __x) __attribute__ ((__nothrow__ )); extern int __ilogb (double __x) __attribute__ ((__nothrow__ ));
# 290 "/usr/include/bits/mathcalls.h" 3 4
extern double scalbln (double __x, long int __n) __attribute__ ((__nothrow__ )); extern double __scalbln (double __x, long int __n) __attribute__ ((__nothrow__ ));



extern double nearbyint (double __x) __attribute__ ((__nothrow__ )); extern double __nearbyint (double __x) __attribute__ ((__nothrow__ ));



extern double round (double __x) __attribute__ ((__nothrow__ )) __attribute__ ((__const__)); extern double __round (double __x) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));



extern double trunc (double __x) __attribute__ ((__nothrow__ )) __attribute__ ((__const__)); extern double __trunc (double __x) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));




extern double remquo (double __x, double __y, int *__quo) __attribute__ ((__nothrow__ )); extern double __remquo (double __x, double __y, int *__quo) __attribute__ ((__nothrow__ ));






extern long int lrint (double __x) __attribute__ ((__nothrow__ )); extern long int __lrint (double __x) __attribute__ ((__nothrow__ ));
__extension__
extern long long int llrint (double __x) __attribute__ ((__nothrow__ )); extern long long int __llrint (double __x) __attribute__ ((__nothrow__ ));



extern long int lround (double __x) __attribute__ ((__nothrow__ )); extern long int __lround (double __x) __attribute__ ((__nothrow__ ));
__extension__
extern long long int llround (double __x) __attribute__ ((__nothrow__ )); extern long long int __llround (double __x) __attribute__ ((__nothrow__ ));



extern double fdim (double __x, double __y) __attribute__ ((__nothrow__ )); extern double __fdim (double __x, double __y) __attribute__ ((__nothrow__ ));


extern double fmax (double __x, double __y) __attribute__ ((__nothrow__ )) __attribute__ ((__const__)); extern double __fmax (double __x, double __y) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));


extern double fmin (double __x, double __y) __attribute__ ((__nothrow__ )) __attribute__ ((__const__)); extern double __fmin (double __x, double __y) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));


extern double fma (double __x, double __y, double __z) __attribute__ ((__nothrow__ )); extern double __fma (double __x, double __y, double __z) __attribute__ ((__nothrow__ ));
# 396 "/usr/include/bits/mathcalls.h" 3 4
extern double scalb (double __x, double __n) __attribute__ ((__nothrow__ )); extern double __scalb (double __x, double __n) __attribute__ ((__nothrow__ ));
# 291 "/usr/include/math.h" 2 3 4
# 306 "/usr/include/math.h" 3 4
# 1 "/usr/include/bits/mathcalls-helper-functions.h" 1 3 4
# 21 "/usr/include/bits/mathcalls-helper-functions.h" 3 4
extern int __fpclassifyf (float __value) __attribute__ ((__nothrow__ ))
     __attribute__ ((__const__));


extern int __signbitf (float __value) __attribute__ ((__nothrow__ ))
     __attribute__ ((__const__));



extern int __isinff (float __value) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));


extern int __finitef (float __value) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));


extern int __isnanf (float __value) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));


extern int __iseqsigf (float __x, float __y) __attribute__ ((__nothrow__ ));


extern int __issignalingf (float __value) __attribute__ ((__nothrow__ ))
     __attribute__ ((__const__));
# 307 "/usr/include/math.h" 2 3 4
# 1 "/usr/include/bits/mathcalls.h" 1 3 4
# 53 "/usr/include/bits/mathcalls.h" 3 4
extern float acosf (float __x) __attribute__ ((__nothrow__ )); extern float __acosf (float __x) __attribute__ ((__nothrow__ ));

extern float asinf (float __x) __attribute__ ((__nothrow__ )); extern float __asinf (float __x) __attribute__ ((__nothrow__ ));

extern float atanf (float __x) __attribute__ ((__nothrow__ )); extern float __atanf (float __x) __attribute__ ((__nothrow__ ));

extern float atan2f (float __y, float __x) __attribute__ ((__nothrow__ )); extern float __atan2f (float __y, float __x) __attribute__ ((__nothrow__ ));


 extern float cosf (float __x) __attribute__ ((__nothrow__ )); extern float __cosf (float __x) __attribute__ ((__nothrow__ ));

 extern float sinf (float __x) __attribute__ ((__nothrow__ )); extern float __sinf (float __x) __attribute__ ((__nothrow__ ));

extern float tanf (float __x) __attribute__ ((__nothrow__ )); extern float __tanf (float __x) __attribute__ ((__nothrow__ ));




extern float coshf (float __x) __attribute__ ((__nothrow__ )); extern float __coshf (float __x) __attribute__ ((__nothrow__ ));

extern float sinhf (float __x) __attribute__ ((__nothrow__ )); extern float __sinhf (float __x) __attribute__ ((__nothrow__ ));

extern float tanhf (float __x) __attribute__ ((__nothrow__ )); extern float __tanhf (float __x) __attribute__ ((__nothrow__ ));
# 85 "/usr/include/bits/mathcalls.h" 3 4
extern float acoshf (float __x) __attribute__ ((__nothrow__ )); extern float __acoshf (float __x) __attribute__ ((__nothrow__ ));

extern float asinhf (float __x) __attribute__ ((__nothrow__ )); extern float __asinhf (float __x) __attribute__ ((__nothrow__ ));

extern float atanhf (float __x) __attribute__ ((__nothrow__ )); extern float __atanhf (float __x) __attribute__ ((__nothrow__ ));





 extern float expf (float __x) __attribute__ ((__nothrow__ )); extern float __expf (float __x) __attribute__ ((__nothrow__ ));


extern float frexpf (float __x, int *__exponent) __attribute__ ((__nothrow__ )); extern float __frexpf (float __x, int *__exponent) __attribute__ ((__nothrow__ ));


extern float ldexpf (float __x, int __exponent) __attribute__ ((__nothrow__ )); extern float __ldexpf (float __x, int __exponent) __attribute__ ((__nothrow__ ));


 extern float logf (float __x) __attribute__ ((__nothrow__ )); extern float __logf (float __x) __attribute__ ((__nothrow__ ));


extern float log10f (float __x) __attribute__ ((__nothrow__ )); extern float __log10f (float __x) __attribute__ ((__nothrow__ ));


extern float modff (float __x, float *__iptr) __attribute__ ((__nothrow__ )); extern float __modff (float __x, float *__iptr) __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (2)));
# 119 "/usr/include/bits/mathcalls.h" 3 4
extern float expm1f (float __x) __attribute__ ((__nothrow__ )); extern float __expm1f (float __x) __attribute__ ((__nothrow__ ));


extern float log1pf (float __x) __attribute__ ((__nothrow__ )); extern float __log1pf (float __x) __attribute__ ((__nothrow__ ));


extern float logbf (float __x) __attribute__ ((__nothrow__ )); extern float __logbf (float __x) __attribute__ ((__nothrow__ ));




extern float exp2f (float __x) __attribute__ ((__nothrow__ )); extern float __exp2f (float __x) __attribute__ ((__nothrow__ ));


extern float log2f (float __x) __attribute__ ((__nothrow__ )); extern float __log2f (float __x) __attribute__ ((__nothrow__ ));






 extern float powf (float __x, float __y) __attribute__ ((__nothrow__ )); extern float __powf (float __x, float __y) __attribute__ ((__nothrow__ ));


extern float sqrtf (float __x) __attribute__ ((__nothrow__ )); extern float __sqrtf (float __x) __attribute__ ((__nothrow__ ));



extern float hypotf (float __x, float __y) __attribute__ ((__nothrow__ )); extern float __hypotf (float __x, float __y) __attribute__ ((__nothrow__ ));




extern float cbrtf (float __x) __attribute__ ((__nothrow__ )); extern float __cbrtf (float __x) __attribute__ ((__nothrow__ ));






extern float ceilf (float __x) __attribute__ ((__nothrow__ )) __attribute__ ((__const__)); extern float __ceilf (float __x) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));


extern float fabsf (float __x) __attribute__ ((__nothrow__ )) __attribute__ ((__const__)); extern float __fabsf (float __x) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));


extern float floorf (float __x) __attribute__ ((__nothrow__ )) __attribute__ ((__const__)); extern float __floorf (float __x) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));


extern float fmodf (float __x, float __y) __attribute__ ((__nothrow__ )); extern float __fmodf (float __x, float __y) __attribute__ ((__nothrow__ ));
# 177 "/usr/include/bits/mathcalls.h" 3 4
extern int isinff (float __value) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));




extern int finitef (float __value) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));


extern float dremf (float __x, float __y) __attribute__ ((__nothrow__ )); extern float __dremf (float __x, float __y) __attribute__ ((__nothrow__ ));



extern float significandf (float __x) __attribute__ ((__nothrow__ )); extern float __significandf (float __x) __attribute__ ((__nothrow__ ));






extern float copysignf (float __x, float __y) __attribute__ ((__nothrow__ )) __attribute__ ((__const__)); extern float __copysignf (float __x, float __y) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));




extern float nanf (const char *__tagb) __attribute__ ((__nothrow__ )); extern float __nanf (const char *__tagb) __attribute__ ((__nothrow__ ));
# 211 "/usr/include/bits/mathcalls.h" 3 4
extern int isnanf (float __value) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));





extern float j0f (float) __attribute__ ((__nothrow__ )); extern float __j0f (float) __attribute__ ((__nothrow__ ));
extern float j1f (float) __attribute__ ((__nothrow__ )); extern float __j1f (float) __attribute__ ((__nothrow__ ));
extern float jnf (int, float) __attribute__ ((__nothrow__ )); extern float __jnf (int, float) __attribute__ ((__nothrow__ ));
extern float y0f (float) __attribute__ ((__nothrow__ )); extern float __y0f (float) __attribute__ ((__nothrow__ ));
extern float y1f (float) __attribute__ ((__nothrow__ )); extern float __y1f (float) __attribute__ ((__nothrow__ ));
extern float ynf (int, float) __attribute__ ((__nothrow__ )); extern float __ynf (int, float) __attribute__ ((__nothrow__ ));





extern float erff (float) __attribute__ ((__nothrow__ )); extern float __erff (float) __attribute__ ((__nothrow__ ));
extern float erfcf (float) __attribute__ ((__nothrow__ )); extern float __erfcf (float) __attribute__ ((__nothrow__ ));
extern float lgammaf (float) __attribute__ ((__nothrow__ )); extern float __lgammaf (float) __attribute__ ((__nothrow__ ));




extern float tgammaf (float) __attribute__ ((__nothrow__ )); extern float __tgammaf (float) __attribute__ ((__nothrow__ ));





extern float gammaf (float) __attribute__ ((__nothrow__ )); extern float __gammaf (float) __attribute__ ((__nothrow__ ));







extern float lgammaf_r (float, int *__signgamp) __attribute__ ((__nothrow__ )); extern float __lgammaf_r (float, int *__signgamp) __attribute__ ((__nothrow__ ));






extern float rintf (float __x) __attribute__ ((__nothrow__ )); extern float __rintf (float __x) __attribute__ ((__nothrow__ ));


extern float nextafterf (float __x, float __y) __attribute__ ((__nothrow__ )); extern float __nextafterf (float __x, float __y) __attribute__ ((__nothrow__ ));

extern float nexttowardf (float __x, long double __y) __attribute__ ((__nothrow__ )); extern float __nexttowardf (float __x, long double __y) __attribute__ ((__nothrow__ ));
# 272 "/usr/include/bits/mathcalls.h" 3 4
extern float remainderf (float __x, float __y) __attribute__ ((__nothrow__ )); extern float __remainderf (float __x, float __y) __attribute__ ((__nothrow__ ));



extern float scalbnf (float __x, int __n) __attribute__ ((__nothrow__ )); extern float __scalbnf (float __x, int __n) __attribute__ ((__nothrow__ ));



extern int ilogbf (float __x) __attribute__ ((__nothrow__ )); extern int __ilogbf (float __x) __attribute__ ((__nothrow__ ));
# 290 "/usr/include/bits/mathcalls.h" 3 4
extern float scalblnf (float __x, long int __n) __attribute__ ((__nothrow__ )); extern float __scalblnf (float __x, long int __n) __attribute__ ((__nothrow__ ));



extern float nearbyintf (float __x) __attribute__ ((__nothrow__ )); extern float __nearbyintf (float __x) __attribute__ ((__nothrow__ ));



extern float roundf (float __x) __attribute__ ((__nothrow__ )) __attribute__ ((__const__)); extern float __roundf (float __x) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));



extern float truncf (float __x) __attribute__ ((__nothrow__ )) __attribute__ ((__const__)); extern float __truncf (float __x) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));




extern float remquof (float __x, float __y, int *__quo) __attribute__ ((__nothrow__ )); extern float __remquof (float __x, float __y, int *__quo) __attribute__ ((__nothrow__ ));






extern long int lrintf (float __x) __attribute__ ((__nothrow__ )); extern long int __lrintf (float __x) __attribute__ ((__nothrow__ ));
__extension__
extern long long int llrintf (float __x) __attribute__ ((__nothrow__ )); extern long long int __llrintf (float __x) __attribute__ ((__nothrow__ ));



extern long int lroundf (float __x) __attribute__ ((__nothrow__ )); extern long int __lroundf (float __x) __attribute__ ((__nothrow__ ));
__extension__
extern long long int llroundf (float __x) __attribute__ ((__nothrow__ )); extern long long int __llroundf (float __x) __attribute__ ((__nothrow__ ));



extern float fdimf (float __x, float __y) __attribute__ ((__nothrow__ )); extern float __fdimf (float __x, float __y) __attribute__ ((__nothrow__ ));


extern float fmaxf (float __x, float __y) __attribute__ ((__nothrow__ )) __attribute__ ((__const__)); extern float __fmaxf (float __x, float __y) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));


extern float fminf (float __x, float __y) __attribute__ ((__nothrow__ )) __attribute__ ((__const__)); extern float __fminf (float __x, float __y) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));


extern float fmaf (float __x, float __y, float __z) __attribute__ ((__nothrow__ )); extern float __fmaf (float __x, float __y, float __z) __attribute__ ((__nothrow__ ));
# 396 "/usr/include/bits/mathcalls.h" 3 4
extern float scalbf (float __x, float __n) __attribute__ ((__nothrow__ )); extern float __scalbf (float __x, float __n) __attribute__ ((__nothrow__ ));
# 308 "/usr/include/math.h" 2 3 4
# 349 "/usr/include/math.h" 3 4
# 1 "/usr/include/bits/mathcalls-helper-functions.h" 1 3 4
# 21 "/usr/include/bits/mathcalls-helper-functions.h" 3 4
extern int __fpclassifyl (long double __value) __attribute__ ((__nothrow__ ))
     __attribute__ ((__const__));


extern int __signbitl (long double __value) __attribute__ ((__nothrow__ ))
     __attribute__ ((__const__));



extern int __isinfl (long double __value) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));


extern int __finitel (long double __value) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));


extern int __isnanl (long double __value) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));


extern int __iseqsigl (long double __x, long double __y) __attribute__ ((__nothrow__ ));


extern int __issignalingl (long double __value) __attribute__ ((__nothrow__ ))
     __attribute__ ((__const__));
# 350 "/usr/include/math.h" 2 3 4
# 1 "/usr/include/bits/mathcalls.h" 1 3 4
# 53 "/usr/include/bits/mathcalls.h" 3 4
extern long double acosl (long double __x) __attribute__ ((__nothrow__ )); extern long double __acosl (long double __x) __attribute__ ((__nothrow__ ));

extern long double asinl (long double __x) __attribute__ ((__nothrow__ )); extern long double __asinl (long double __x) __attribute__ ((__nothrow__ ));

extern long double atanl (long double __x) __attribute__ ((__nothrow__ )); extern long double __atanl (long double __x) __attribute__ ((__nothrow__ ));

extern long double atan2l (long double __y, long double __x) __attribute__ ((__nothrow__ )); extern long double __atan2l (long double __y, long double __x) __attribute__ ((__nothrow__ ));


 extern long double cosl (long double __x) __attribute__ ((__nothrow__ )); extern long double __cosl (long double __x) __attribute__ ((__nothrow__ ));

 extern long double sinl (long double __x) __attribute__ ((__nothrow__ )); extern long double __sinl (long double __x) __attribute__ ((__nothrow__ ));

extern long double tanl (long double __x) __attribute__ ((__nothrow__ )); extern long double __tanl (long double __x) __attribute__ ((__nothrow__ ));




extern long double coshl (long double __x) __attribute__ ((__nothrow__ )); extern long double __coshl (long double __x) __attribute__ ((__nothrow__ ));

extern long double sinhl (long double __x) __attribute__ ((__nothrow__ )); extern long double __sinhl (long double __x) __attribute__ ((__nothrow__ ));

extern long double tanhl (long double __x) __attribute__ ((__nothrow__ )); extern long double __tanhl (long double __x) __attribute__ ((__nothrow__ ));
# 85 "/usr/include/bits/mathcalls.h" 3 4
extern long double acoshl (long double __x) __attribute__ ((__nothrow__ )); extern long double __acoshl (long double __x) __attribute__ ((__nothrow__ ));

extern long double asinhl (long double __x) __attribute__ ((__nothrow__ )); extern long double __asinhl (long double __x) __attribute__ ((__nothrow__ ));

extern long double atanhl (long double __x) __attribute__ ((__nothrow__ )); extern long double __atanhl (long double __x) __attribute__ ((__nothrow__ ));





 extern long double expl (long double __x) __attribute__ ((__nothrow__ )); extern long double __expl (long double __x) __attribute__ ((__nothrow__ ));


extern long double frexpl (long double __x, int *__exponent) __attribute__ ((__nothrow__ )); extern long double __frexpl (long double __x, int *__exponent) __attribute__ ((__nothrow__ ));


extern long double ldexpl (long double __x, int __exponent) __attribute__ ((__nothrow__ )); extern long double __ldexpl (long double __x, int __exponent) __attribute__ ((__nothrow__ ));


 extern long double logl (long double __x) __attribute__ ((__nothrow__ )); extern long double __logl (long double __x) __attribute__ ((__nothrow__ ));


extern long double log10l (long double __x) __attribute__ ((__nothrow__ )); extern long double __log10l (long double __x) __attribute__ ((__nothrow__ ));


extern long double modfl (long double __x, long double *__iptr) __attribute__ ((__nothrow__ )); extern long double __modfl (long double __x, long double *__iptr) __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (2)));
# 119 "/usr/include/bits/mathcalls.h" 3 4
extern long double expm1l (long double __x) __attribute__ ((__nothrow__ )); extern long double __expm1l (long double __x) __attribute__ ((__nothrow__ ));


extern long double log1pl (long double __x) __attribute__ ((__nothrow__ )); extern long double __log1pl (long double __x) __attribute__ ((__nothrow__ ));


extern long double logbl (long double __x) __attribute__ ((__nothrow__ )); extern long double __logbl (long double __x) __attribute__ ((__nothrow__ ));




extern long double exp2l (long double __x) __attribute__ ((__nothrow__ )); extern long double __exp2l (long double __x) __attribute__ ((__nothrow__ ));


extern long double log2l (long double __x) __attribute__ ((__nothrow__ )); extern long double __log2l (long double __x) __attribute__ ((__nothrow__ ));






 extern long double powl (long double __x, long double __y) __attribute__ ((__nothrow__ )); extern long double __powl (long double __x, long double __y) __attribute__ ((__nothrow__ ));


extern long double sqrtl (long double __x) __attribute__ ((__nothrow__ )); extern long double __sqrtl (long double __x) __attribute__ ((__nothrow__ ));



extern long double hypotl (long double __x, long double __y) __attribute__ ((__nothrow__ )); extern long double __hypotl (long double __x, long double __y) __attribute__ ((__nothrow__ ));




extern long double cbrtl (long double __x) __attribute__ ((__nothrow__ )); extern long double __cbrtl (long double __x) __attribute__ ((__nothrow__ ));






extern long double ceill (long double __x) __attribute__ ((__nothrow__ )) __attribute__ ((__const__)); extern long double __ceill (long double __x) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));


extern long double fabsl (long double __x) __attribute__ ((__nothrow__ )) __attribute__ ((__const__)); extern long double __fabsl (long double __x) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));


extern long double floorl (long double __x) __attribute__ ((__nothrow__ )) __attribute__ ((__const__)); extern long double __floorl (long double __x) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));


extern long double fmodl (long double __x, long double __y) __attribute__ ((__nothrow__ )); extern long double __fmodl (long double __x, long double __y) __attribute__ ((__nothrow__ ));
# 177 "/usr/include/bits/mathcalls.h" 3 4
extern int isinfl (long double __value) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));




extern int finitel (long double __value) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));


extern long double dreml (long double __x, long double __y) __attribute__ ((__nothrow__ )); extern long double __dreml (long double __x, long double __y) __attribute__ ((__nothrow__ ));



extern long double significandl (long double __x) __attribute__ ((__nothrow__ )); extern long double __significandl (long double __x) __attribute__ ((__nothrow__ ));






extern long double copysignl (long double __x, long double __y) __attribute__ ((__nothrow__ )) __attribute__ ((__const__)); extern long double __copysignl (long double __x, long double __y) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));




extern long double nanl (const char *__tagb) __attribute__ ((__nothrow__ )); extern long double __nanl (const char *__tagb) __attribute__ ((__nothrow__ ));
# 211 "/usr/include/bits/mathcalls.h" 3 4
extern int isnanl (long double __value) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));





extern long double j0l (long double) __attribute__ ((__nothrow__ )); extern long double __j0l (long double) __attribute__ ((__nothrow__ ));
extern long double j1l (long double) __attribute__ ((__nothrow__ )); extern long double __j1l (long double) __attribute__ ((__nothrow__ ));
extern long double jnl (int, long double) __attribute__ ((__nothrow__ )); extern long double __jnl (int, long double) __attribute__ ((__nothrow__ ));
extern long double y0l (long double) __attribute__ ((__nothrow__ )); extern long double __y0l (long double) __attribute__ ((__nothrow__ ));
extern long double y1l (long double) __attribute__ ((__nothrow__ )); extern long double __y1l (long double) __attribute__ ((__nothrow__ ));
extern long double ynl (int, long double) __attribute__ ((__nothrow__ )); extern long double __ynl (int, long double) __attribute__ ((__nothrow__ ));





extern long double erfl (long double) __attribute__ ((__nothrow__ )); extern long double __erfl (long double) __attribute__ ((__nothrow__ ));
extern long double erfcl (long double) __attribute__ ((__nothrow__ )); extern long double __erfcl (long double) __attribute__ ((__nothrow__ ));
extern long double lgammal (long double) __attribute__ ((__nothrow__ )); extern long double __lgammal (long double) __attribute__ ((__nothrow__ ));




extern long double tgammal (long double) __attribute__ ((__nothrow__ )); extern long double __tgammal (long double) __attribute__ ((__nothrow__ ));





extern long double gammal (long double) __attribute__ ((__nothrow__ )); extern long double __gammal (long double) __attribute__ ((__nothrow__ ));







extern long double lgammal_r (long double, int *__signgamp) __attribute__ ((__nothrow__ )); extern long double __lgammal_r (long double, int *__signgamp) __attribute__ ((__nothrow__ ));






extern long double rintl (long double __x) __attribute__ ((__nothrow__ )); extern long double __rintl (long double __x) __attribute__ ((__nothrow__ ));


extern long double nextafterl (long double __x, long double __y) __attribute__ ((__nothrow__ )); extern long double __nextafterl (long double __x, long double __y) __attribute__ ((__nothrow__ ));

extern long double nexttowardl (long double __x, long double __y) __attribute__ ((__nothrow__ )); extern long double __nexttowardl (long double __x, long double __y) __attribute__ ((__nothrow__ ));
# 272 "/usr/include/bits/mathcalls.h" 3 4
extern long double remainderl (long double __x, long double __y) __attribute__ ((__nothrow__ )); extern long double __remainderl (long double __x, long double __y) __attribute__ ((__nothrow__ ));



extern long double scalbnl (long double __x, int __n) __attribute__ ((__nothrow__ )); extern long double __scalbnl (long double __x, int __n) __attribute__ ((__nothrow__ ));



extern int ilogbl (long double __x) __attribute__ ((__nothrow__ )); extern int __ilogbl (long double __x) __attribute__ ((__nothrow__ ));
# 290 "/usr/include/bits/mathcalls.h" 3 4
extern long double scalblnl (long double __x, long int __n) __attribute__ ((__nothrow__ )); extern long double __scalblnl (long double __x, long int __n) __attribute__ ((__nothrow__ ));



extern long double nearbyintl (long double __x) __attribute__ ((__nothrow__ )); extern long double __nearbyintl (long double __x) __attribute__ ((__nothrow__ ));



extern long double roundl (long double __x) __attribute__ ((__nothrow__ )) __attribute__ ((__const__)); extern long double __roundl (long double __x) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));



extern long double truncl (long double __x) __attribute__ ((__nothrow__ )) __attribute__ ((__const__)); extern long double __truncl (long double __x) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));




extern long double remquol (long double __x, long double __y, int *__quo) __attribute__ ((__nothrow__ )); extern long double __remquol (long double __x, long double __y, int *__quo) __attribute__ ((__nothrow__ ));






extern long int lrintl (long double __x) __attribute__ ((__nothrow__ )); extern long int __lrintl (long double __x) __attribute__ ((__nothrow__ ));
__extension__
extern long long int llrintl (long double __x) __attribute__ ((__nothrow__ )); extern long long int __llrintl (long double __x) __attribute__ ((__nothrow__ ));



extern long int lroundl (long double __x) __attribute__ ((__nothrow__ )); extern long int __lroundl (long double __x) __attribute__ ((__nothrow__ ));
__extension__
extern long long int llroundl (long double __x) __attribute__ ((__nothrow__ )); extern long long int __llroundl (long double __x) __attribute__ ((__nothrow__ ));



extern long double fdiml (long double __x, long double __y) __attribute__ ((__nothrow__ )); extern long double __fdiml (long double __x, long double __y) __attribute__ ((__nothrow__ ));


extern long double fmaxl (long double __x, long double __y) __attribute__ ((__nothrow__ )) __attribute__ ((__const__)); extern long double __fmaxl (long double __x, long double __y) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));


extern long double fminl (long double __x, long double __y) __attribute__ ((__nothrow__ )) __attribute__ ((__const__)); extern long double __fminl (long double __x, long double __y) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));


extern long double fmal (long double __x, long double __y, long double __z) __attribute__ ((__nothrow__ )); extern long double __fmal (long double __x, long double __y, long double __z) __attribute__ ((__nothrow__ ));
# 396 "/usr/include/bits/mathcalls.h" 3 4
extern long double scalbl (long double __x, long double __n) __attribute__ ((__nothrow__ )); extern long double __scalbl (long double __x, long double __n) __attribute__ ((__nothrow__ ));
# 351 "/usr/include/math.h" 2 3 4
# 773 "/usr/include/math.h" 3 4
extern int signgam;
# 853 "/usr/include/math.h" 3 4
enum
  {
    FP_NAN =

      0,
    FP_INFINITE =

      1,
    FP_ZERO =

      2,
    FP_SUBNORMAL =

      3,
    FP_NORMAL =

      4
  };
# 35 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/openmp_wrappers/math.h" 2 3



# 1 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/limits.h" 1 3
# 21 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/limits.h" 3
# 1 "/usr/include/limits.h" 1 3 4
# 26 "/usr/include/limits.h" 3 4
# 1 "/usr/include/bits/libc-header-start.h" 1 3 4
# 27 "/usr/include/limits.h" 2 3 4
# 183 "/usr/include/limits.h" 3 4
# 1 "/usr/include/bits/posix1_lim.h" 1 3 4
# 27 "/usr/include/bits/posix1_lim.h" 3 4
# 1 "/usr/include/bits/wordsize.h" 1 3 4
# 28 "/usr/include/bits/posix1_lim.h" 2 3 4
# 161 "/usr/include/bits/posix1_lim.h" 3 4
# 1 "/usr/include/bits/local_lim.h" 1 3 4
# 38 "/usr/include/bits/local_lim.h" 3 4
# 1 "/usr/include/linux/limits.h" 1 3 4
# 39 "/usr/include/bits/local_lim.h" 2 3 4
# 162 "/usr/include/bits/posix1_lim.h" 2 3 4
# 184 "/usr/include/limits.h" 2 3 4



# 1 "/usr/include/bits/posix2_lim.h" 1 3 4
# 188 "/usr/include/limits.h" 2 3 4
# 22 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/limits.h" 2 3
# 39 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/openmp_wrappers/math.h" 2 3



# 1 "/usr/include/stdlib.h" 1 3 4
# 25 "/usr/include/stdlib.h" 3 4
# 1 "/usr/include/bits/libc-header-start.h" 1 3 4
# 26 "/usr/include/stdlib.h" 2 3 4





# 1 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/stddef.h" 1 3 4
# 74 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/stddef.h" 3 4
typedef int wchar_t;
# 32 "/usr/include/stdlib.h" 2 3 4







# 1 "/usr/include/bits/waitflags.h" 1 3 4
# 40 "/usr/include/stdlib.h" 2 3 4
# 1 "/usr/include/bits/waitstatus.h" 1 3 4
# 41 "/usr/include/stdlib.h" 2 3 4
# 58 "/usr/include/stdlib.h" 3 4
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
# 97 "/usr/include/sys/types.h" 3 4
typedef __pid_t pid_t;





typedef __id_t id_t;
# 114 "/usr/include/sys/types.h" 3 4
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
# 43 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/openmp_wrappers/math.h" 2 3

#pragma omp begin declare variant match( device = {arch(nvptx, nvptx64)}, implementation = {extension(match_any)})





# 1 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/__clang_cuda_math.h" 1 3
# 66 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/__clang_cuda_math.h" 3
static __attribute__((always_inline, nothrow)) int abs(int __a) { return __nv_abs(__a); }
static __attribute__((always_inline, nothrow)) double fabs(double __a) { return __nv_fabs(__a); }
static __attribute__((always_inline, nothrow)) double acos(double __a) { return __nv_acos(__a); }
static __attribute__((always_inline, nothrow)) float acosf(float __a) { return __nv_acosf(__a); }
static __attribute__((always_inline, nothrow)) double acosh(double __a) { return __nv_acosh(__a); }
static __attribute__((always_inline, nothrow)) float acoshf(float __a) { return __nv_acoshf(__a); }
static __attribute__((always_inline, nothrow)) double asin(double __a) { return __nv_asin(__a); }
static __attribute__((always_inline, nothrow)) float asinf(float __a) { return __nv_asinf(__a); }
static __attribute__((always_inline, nothrow)) double asinh(double __a) { return __nv_asinh(__a); }
static __attribute__((always_inline, nothrow)) float asinhf(float __a) { return __nv_asinhf(__a); }
static __attribute__((always_inline, nothrow)) double atan(double __a) { return __nv_atan(__a); }
static __attribute__((always_inline, nothrow)) double atan2(double __a, double __b) { return __nv_atan2(__a, __b); }
static __attribute__((always_inline, nothrow)) float atan2f(float __a, float __b) { return __nv_atan2f(__a, __b); }
static __attribute__((always_inline, nothrow)) float atanf(float __a) { return __nv_atanf(__a); }
static __attribute__((always_inline, nothrow)) double atanh(double __a) { return __nv_atanh(__a); }
static __attribute__((always_inline, nothrow)) float atanhf(float __a) { return __nv_atanhf(__a); }
static __attribute__((always_inline, nothrow)) double cbrt(double __a) { return __nv_cbrt(__a); }
static __attribute__((always_inline, nothrow)) float cbrtf(float __a) { return __nv_cbrtf(__a); }
static __attribute__((always_inline, nothrow)) double ceil(double __a) { return __nv_ceil(__a); }
static __attribute__((always_inline, nothrow)) float ceilf(float __a) { return __nv_ceilf(__a); }
static __attribute__((always_inline, nothrow)) double copysign(double __a, double __b) {
  return __nv_copysign(__a, __b);
}
static __attribute__((always_inline, nothrow)) float copysignf(float __a, float __b) {
  return __nv_copysignf(__a, __b);
}
static __attribute__((always_inline, nothrow)) double cos(double __a) { return __nv_cos(__a); }
static __attribute__((always_inline, nothrow)) float cosf(float __a) {
  return __nv_cosf(__a);
}
static __attribute__((always_inline, nothrow)) double cosh(double __a) { return __nv_cosh(__a); }
static __attribute__((always_inline, nothrow)) float coshf(float __a) { return __nv_coshf(__a); }
static __attribute__((always_inline, nothrow)) double cospi(double __a) { return __nv_cospi(__a); }
static __attribute__((always_inline, nothrow)) float cospif(float __a) { return __nv_cospif(__a); }
static __attribute__((always_inline, nothrow)) double cyl_bessel_i0(double __a) { return __nv_cyl_bessel_i0(__a); }
static __attribute__((always_inline, nothrow)) float cyl_bessel_i0f(float __a) { return __nv_cyl_bessel_i0f(__a); }
static __attribute__((always_inline, nothrow)) double cyl_bessel_i1(double __a) { return __nv_cyl_bessel_i1(__a); }
static __attribute__((always_inline, nothrow)) float cyl_bessel_i1f(float __a) { return __nv_cyl_bessel_i1f(__a); }
static __attribute__((always_inline, nothrow)) double erf(double __a) { return __nv_erf(__a); }
static __attribute__((always_inline, nothrow)) double erfc(double __a) { return __nv_erfc(__a); }
static __attribute__((always_inline, nothrow)) float erfcf(float __a) { return __nv_erfcf(__a); }
static __attribute__((always_inline, nothrow)) double erfcinv(double __a) { return __nv_erfcinv(__a); }
static __attribute__((always_inline, nothrow)) float erfcinvf(float __a) { return __nv_erfcinvf(__a); }
static __attribute__((always_inline, nothrow)) double erfcx(double __a) { return __nv_erfcx(__a); }
static __attribute__((always_inline, nothrow)) float erfcxf(float __a) { return __nv_erfcxf(__a); }
static __attribute__((always_inline, nothrow)) float erff(float __a) { return __nv_erff(__a); }
static __attribute__((always_inline, nothrow)) double erfinv(double __a) { return __nv_erfinv(__a); }
static __attribute__((always_inline, nothrow)) float erfinvf(float __a) { return __nv_erfinvf(__a); }
static __attribute__((always_inline, nothrow)) double exp(double __a) { return __nv_exp(__a); }
static __attribute__((always_inline, nothrow)) double exp10(double __a) { return __nv_exp10(__a); }
static __attribute__((always_inline, nothrow)) float exp10f(float __a) { return __nv_exp10f(__a); }
static __attribute__((always_inline, nothrow)) double exp2(double __a) { return __nv_exp2(__a); }
static __attribute__((always_inline, nothrow)) float exp2f(float __a) { return __nv_exp2f(__a); }
static __attribute__((always_inline, nothrow)) float expf(float __a) { return __nv_expf(__a); }
static __attribute__((always_inline, nothrow)) double expm1(double __a) { return __nv_expm1(__a); }
static __attribute__((always_inline, nothrow)) float expm1f(float __a) { return __nv_expm1f(__a); }
static __attribute__((always_inline, nothrow)) float fabsf(float __a) { return __nv_fabsf(__a); }
static __attribute__((always_inline, nothrow)) double fdim(double __a, double __b) { return __nv_fdim(__a, __b); }
static __attribute__((always_inline, nothrow)) float fdimf(float __a, float __b) { return __nv_fdimf(__a, __b); }
static __attribute__((always_inline, nothrow)) double fdivide(double __a, double __b) { return __a / __b; }
static __attribute__((always_inline, nothrow)) float fdividef(float __a, float __b) {



  return __a / __b;

}
static __attribute__((always_inline, nothrow)) double floor(double __f) { return __nv_floor(__f); }
static __attribute__((always_inline, nothrow)) float floorf(float __f) { return __nv_floorf(__f); }
static __attribute__((always_inline, nothrow)) double fma(double __a, double __b, double __c) {
  return __nv_fma(__a, __b, __c);
}
static __attribute__((always_inline, nothrow)) float fmaf(float __a, float __b, float __c) {
  return __nv_fmaf(__a, __b, __c);
}
static __attribute__((always_inline, nothrow)) double fmax(double __a, double __b) { return __nv_fmax(__a, __b); }
static __attribute__((always_inline, nothrow)) float fmaxf(float __a, float __b) { return __nv_fmaxf(__a, __b); }
static __attribute__((always_inline, nothrow)) double fmin(double __a, double __b) { return __nv_fmin(__a, __b); }
static __attribute__((always_inline, nothrow)) float fminf(float __a, float __b) { return __nv_fminf(__a, __b); }
static __attribute__((always_inline, nothrow)) double fmod(double __a, double __b) { return __nv_fmod(__a, __b); }
static __attribute__((always_inline, nothrow)) float fmodf(float __a, float __b) { return __nv_fmodf(__a, __b); }
static __attribute__((always_inline, nothrow)) double frexp(double __a, int *__b) { return __nv_frexp(__a, __b); }
static __attribute__((always_inline, nothrow)) float frexpf(float __a, int *__b) { return __nv_frexpf(__a, __b); }
static __attribute__((always_inline, nothrow)) double hypot(double __a, double __b) { return __nv_hypot(__a, __b); }
static __attribute__((always_inline, nothrow)) float hypotf(float __a, float __b) { return __nv_hypotf(__a, __b); }
static __attribute__((always_inline, nothrow)) int ilogb(double __a) { return __nv_ilogb(__a); }
static __attribute__((always_inline, nothrow)) int ilogbf(float __a) { return __nv_ilogbf(__a); }
static __attribute__((always_inline, nothrow)) double j0(double __a) { return __nv_j0(__a); }
static __attribute__((always_inline, nothrow)) float j0f(float __a) { return __nv_j0f(__a); }
static __attribute__((always_inline, nothrow)) double j1(double __a) { return __nv_j1(__a); }
static __attribute__((always_inline, nothrow)) float j1f(float __a) { return __nv_j1f(__a); }
static __attribute__((always_inline, nothrow)) double jn(int __n, double __a) { return __nv_jn(__n, __a); }
static __attribute__((always_inline, nothrow)) float jnf(int __n, float __a) { return __nv_jnf(__n, __a); }

static __attribute__((always_inline, nothrow)) long labs(long __a) { return __nv_llabs(__a); };



static __attribute__((always_inline, nothrow)) double ldexp(double __a, int __b) { return __nv_ldexp(__a, __b); }
static __attribute__((always_inline, nothrow)) float ldexpf(float __a, int __b) { return __nv_ldexpf(__a, __b); }
static __attribute__((always_inline, nothrow)) double lgamma(double __a) { return __nv_lgamma(__a); }
static __attribute__((always_inline, nothrow)) float lgammaf(float __a) { return __nv_lgammaf(__a); }
static __attribute__((always_inline, nothrow)) long long llabs(long long __a) { return __nv_llabs(__a); }
static __attribute__((always_inline, nothrow)) long long llmax(long long __a, long long __b) {
  return __nv_llmax(__a, __b);
}
static __attribute__((always_inline, nothrow)) long long llmin(long long __a, long long __b) {
  return __nv_llmin(__a, __b);
}
static __attribute__((always_inline, nothrow)) long long llrint(double __a) { return __nv_llrint(__a); }
static __attribute__((always_inline, nothrow)) long long llrintf(float __a) { return __nv_llrintf(__a); }
static __attribute__((always_inline, nothrow)) long long llround(double __a) { return __nv_llround(__a); }
static __attribute__((always_inline, nothrow)) long long llroundf(float __a) { return __nv_llroundf(__a); }
static __attribute__((always_inline, nothrow)) double log(double __a) { return __nv_log(__a); }
static __attribute__((always_inline, nothrow)) double log10(double __a) { return __nv_log10(__a); }
static __attribute__((always_inline, nothrow)) float log10f(float __a) { return __nv_log10f(__a); }
static __attribute__((always_inline, nothrow)) double log1p(double __a) { return __nv_log1p(__a); }
static __attribute__((always_inline, nothrow)) float log1pf(float __a) { return __nv_log1pf(__a); }
static __attribute__((always_inline, nothrow)) double log2(double __a) { return __nv_log2(__a); }
static __attribute__((always_inline, nothrow)) float log2f(float __a) {
  return __nv_log2f(__a);
}
static __attribute__((always_inline, nothrow)) double logb(double __a) { return __nv_logb(__a); }
static __attribute__((always_inline, nothrow)) float logbf(float __a) { return __nv_logbf(__a); }
static __attribute__((always_inline, nothrow)) float logf(float __a) {
  return __nv_logf(__a);
}

static __attribute__((always_inline, nothrow)) long lrint(double __a) { return llrint(__a); }
static __attribute__((always_inline, nothrow)) long lrintf(float __a) { return __float2ll_rn(__a); }
static __attribute__((always_inline, nothrow)) long lround(double __a) { return llround(__a); }
static __attribute__((always_inline, nothrow)) long lroundf(float __a) { return llroundf(__a); }






static __attribute__((always_inline, nothrow)) int max(int __a, int __b) { return __nv_max(__a, __b); }
static __attribute__((always_inline, nothrow)) int min(int __a, int __b) { return __nv_min(__a, __b); }
static __attribute__((always_inline, nothrow)) double modf(double __a, double *__b) { return __nv_modf(__a, __b); }
static __attribute__((always_inline, nothrow)) float modff(float __a, float *__b) { return __nv_modff(__a, __b); }
static __attribute__((always_inline, nothrow)) double nearbyint(double __a) { return __builtin_nearbyint(__a); }
static __attribute__((always_inline, nothrow)) float nearbyintf(float __a) { return __builtin_nearbyintf(__a); }
static __attribute__((always_inline, nothrow)) double nextafter(double __a, double __b) {
  return __nv_nextafter(__a, __b);
}
static __attribute__((always_inline, nothrow)) float nextafterf(float __a, float __b) {
  return __nv_nextafterf(__a, __b);
}
static __attribute__((always_inline, nothrow)) double norm(int __dim, const double *__t) {
  return __nv_norm(__dim, __t);
}
static __attribute__((always_inline, nothrow)) double norm3d(double __a, double __b, double __c) {
  return __nv_norm3d(__a, __b, __c);
}
static __attribute__((always_inline, nothrow)) float norm3df(float __a, float __b, float __c) {
  return __nv_norm3df(__a, __b, __c);
}
static __attribute__((always_inline, nothrow)) double norm4d(double __a, double __b, double __c, double __d) {
  return __nv_norm4d(__a, __b, __c, __d);
}
static __attribute__((always_inline, nothrow)) float norm4df(float __a, float __b, float __c, float __d) {
  return __nv_norm4df(__a, __b, __c, __d);
}
static __attribute__((always_inline, nothrow)) double normcdf(double __a) { return __nv_normcdf(__a); }
static __attribute__((always_inline, nothrow)) float normcdff(float __a) { return __nv_normcdff(__a); }
static __attribute__((always_inline, nothrow)) double normcdfinv(double __a) { return __nv_normcdfinv(__a); }
static __attribute__((always_inline, nothrow)) float normcdfinvf(float __a) { return __nv_normcdfinvf(__a); }
static __attribute__((always_inline, nothrow)) float normf(int __dim, const float *__t) {
  return __nv_normf(__dim, __t);
}
static __attribute__((always_inline, nothrow)) double pow(double __a, double __b) { return __nv_pow(__a, __b); }
static __attribute__((always_inline, nothrow)) float powf(float __a, float __b) { return __nv_powf(__a, __b); }
static __attribute__((always_inline, nothrow)) double powi(double __a, int __b) { return __nv_powi(__a, __b); }
static __attribute__((always_inline, nothrow)) float powif(float __a, int __b) { return __nv_powif(__a, __b); }
static __attribute__((always_inline, nothrow)) double rcbrt(double __a) { return __nv_rcbrt(__a); }
static __attribute__((always_inline, nothrow)) float rcbrtf(float __a) { return __nv_rcbrtf(__a); }
static __attribute__((always_inline, nothrow)) double remainder(double __a, double __b) {
  return __nv_remainder(__a, __b);
}
static __attribute__((always_inline, nothrow)) float remainderf(float __a, float __b) {
  return __nv_remainderf(__a, __b);
}
static __attribute__((always_inline, nothrow)) double remquo(double __a, double __b, int *__c) {
  return __nv_remquo(__a, __b, __c);
}
static __attribute__((always_inline, nothrow)) float remquof(float __a, float __b, int *__c) {
  return __nv_remquof(__a, __b, __c);
}
static __attribute__((always_inline, nothrow)) double rhypot(double __a, double __b) {
  return __nv_rhypot(__a, __b);
}
static __attribute__((always_inline, nothrow)) float rhypotf(float __a, float __b) {
  return __nv_rhypotf(__a, __b);
}

static __attribute__((always_inline, nothrow)) double rint(double __a) { return __builtin_rint(__a); }
static __attribute__((always_inline, nothrow)) float rintf(float __a) { return __builtin_rintf(__a); }
static __attribute__((always_inline, nothrow)) double rnorm(int __a, const double *__b) {
  return __nv_rnorm(__a, __b);
}
static __attribute__((always_inline, nothrow)) double rnorm3d(double __a, double __b, double __c) {
  return __nv_rnorm3d(__a, __b, __c);
}
static __attribute__((always_inline, nothrow)) float rnorm3df(float __a, float __b, float __c) {
  return __nv_rnorm3df(__a, __b, __c);
}
static __attribute__((always_inline, nothrow)) double rnorm4d(double __a, double __b, double __c, double __d) {
  return __nv_rnorm4d(__a, __b, __c, __d);
}
static __attribute__((always_inline, nothrow)) float rnorm4df(float __a, float __b, float __c, float __d) {
  return __nv_rnorm4df(__a, __b, __c, __d);
}
static __attribute__((always_inline, nothrow)) float rnormf(int __dim, const float *__t) {
  return __nv_rnormf(__dim, __t);
}
static __attribute__((always_inline, nothrow)) double round(double __a) { return __nv_round(__a); }
static __attribute__((always_inline, nothrow)) float roundf(float __a) { return __nv_roundf(__a); }
static __attribute__((always_inline, nothrow)) double rsqrt(double __a) { return __nv_rsqrt(__a); }
static __attribute__((always_inline, nothrow)) float rsqrtf(float __a) { return __nv_rsqrtf(__a); }
static __attribute__((always_inline, nothrow)) double scalbn(double __a, int __b) { return __nv_scalbn(__a, __b); }
static __attribute__((always_inline, nothrow)) float scalbnf(float __a, int __b) { return __nv_scalbnf(__a, __b); }
static __attribute__((always_inline, nothrow)) double scalbln(double __a, long __b) {
  if (__b > 2147483647)
    return __a > 0 ? (__builtin_huge_val ()) : -(__builtin_huge_val ());
  if (__b < (-2147483647 -1))
    return __a > 0 ? 0.0 : -0.0;
  return scalbn(__a, (int)__b);
}
static __attribute__((always_inline, nothrow)) float scalblnf(float __a, long __b) {
  if (__b > 2147483647)
    return __a > 0 ? (__builtin_huge_valf ()) : -(__builtin_huge_valf ());
  if (__b < (-2147483647 -1))
    return __a > 0 ? 0.f : -0.f;
  return scalbnf(__a, (int)__b);
}
static __attribute__((always_inline, nothrow)) double sin(double __a) { return __nv_sin(__a); }
static __attribute__((always_inline, nothrow)) void sincos(double __a, double *__s, double *__c) {
  return __nv_sincos(__a, __s, __c);
}
static __attribute__((always_inline, nothrow)) void sincosf(float __a, float *__s, float *__c) {
  return __nv_sincosf(__a, __s, __c);
}
static __attribute__((always_inline, nothrow)) void sincospi(double __a, double *__s, double *__c) {
  return __nv_sincospi(__a, __s, __c);
}
static __attribute__((always_inline, nothrow)) void sincospif(float __a, float *__s, float *__c) {
  return __nv_sincospif(__a, __s, __c);
}
static __attribute__((always_inline, nothrow)) float sinf(float __a) {
  return __nv_sinf(__a);
}
static __attribute__((always_inline, nothrow)) double sinh(double __a) { return __nv_sinh(__a); }
static __attribute__((always_inline, nothrow)) float sinhf(float __a) { return __nv_sinhf(__a); }
static __attribute__((always_inline, nothrow)) double sinpi(double __a) { return __nv_sinpi(__a); }
static __attribute__((always_inline, nothrow)) float sinpif(float __a) { return __nv_sinpif(__a); }
static __attribute__((always_inline, nothrow)) double sqrt(double __a) { return __nv_sqrt(__a); }
static __attribute__((always_inline, nothrow)) float sqrtf(float __a) { return __nv_sqrtf(__a); }
static __attribute__((always_inline, nothrow)) double tan(double __a) { return __nv_tan(__a); }
static __attribute__((always_inline, nothrow)) float tanf(float __a) { return __nv_tanf(__a); }
static __attribute__((always_inline, nothrow)) double tanh(double __a) { return __nv_tanh(__a); }
static __attribute__((always_inline, nothrow)) float tanhf(float __a) { return __nv_tanhf(__a); }
static __attribute__((always_inline, nothrow)) double tgamma(double __a) { return __nv_tgamma(__a); }
static __attribute__((always_inline, nothrow)) float tgammaf(float __a) { return __nv_tgammaf(__a); }
static __attribute__((always_inline, nothrow)) double trunc(double __a) { return __nv_trunc(__a); }
static __attribute__((always_inline, nothrow)) float truncf(float __a) { return __nv_truncf(__a); }
static __attribute__((always_inline, nothrow)) unsigned long long ullmax(unsigned long long __a,
                                     unsigned long long __b) {
  return __nv_ullmax(__a, __b);
}
static __attribute__((always_inline, nothrow)) unsigned long long ullmin(unsigned long long __a,
                                     unsigned long long __b) {
  return __nv_ullmin(__a, __b);
}
static __attribute__((always_inline, nothrow)) unsigned int umax(unsigned int __a, unsigned int __b) {
  return __nv_umax(__a, __b);
}
static __attribute__((always_inline, nothrow)) unsigned int umin(unsigned int __a, unsigned int __b) {
  return __nv_umin(__a, __b);
}
static __attribute__((always_inline, nothrow)) double y0(double __a) { return __nv_y0(__a); }
static __attribute__((always_inline, nothrow)) float y0f(float __a) { return __nv_y0f(__a); }
static __attribute__((always_inline, nothrow)) double y1(double __a) { return __nv_y1(__a); }
static __attribute__((always_inline, nothrow)) float y1f(float __a) { return __nv_y1f(__a); }
static __attribute__((always_inline, nothrow)) double yn(int __a, double __b) { return __nv_yn(__a, __b); }
static __attribute__((always_inline, nothrow)) float ynf(int __a, float __b) { return __nv_ynf(__a, __b); }
# 50 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/openmp_wrappers/math.h" 2 3



#pragma omp end declare variant

#pragma omp begin declare variant match( device = {arch(amdgcn)}, implementation = {extension(match_any)})
# 66 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/openmp_wrappers/math.h" 3
# 1 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/__clang_hip_math.h" 1 3
# 89 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/__clang_hip_math.h" 3
static __attribute__((always_inline, nothrow))
uint64_t __make_mantissa_base8(const char *__tagp) {
  uint64_t __r = 0;
  while (__tagp) {
    char __tmp = *__tagp;

    if (__tmp >= '0' && __tmp <= '7')
      __r = (__r * 8u) + __tmp - '0';
    else
      return 0;

    ++__tagp;
  }

  return __r;
}

static __attribute__((always_inline, nothrow))
uint64_t __make_mantissa_base10(const char *__tagp) {
  uint64_t __r = 0;
  while (__tagp) {
    char __tmp = *__tagp;

    if (__tmp >= '0' && __tmp <= '9')
      __r = (__r * 10u) + __tmp - '0';
    else
      return 0;

    ++__tagp;
  }

  return __r;
}

static __attribute__((always_inline, nothrow))
uint64_t __make_mantissa_base16(const char *__tagp) {
  uint64_t __r = 0;
  while (__tagp) {
    char __tmp = *__tagp;

    if (__tmp >= '0' && __tmp <= '9')
      __r = (__r * 16u) + __tmp - '0';
    else if (__tmp >= 'a' && __tmp <= 'f')
      __r = (__r * 16u) + __tmp - 'a' + 10;
    else if (__tmp >= 'A' && __tmp <= 'F')
      __r = (__r * 16u) + __tmp - 'A' + 10;
    else
      return 0;

    ++__tagp;
  }

  return __r;
}

static __attribute__((always_inline, nothrow))
uint64_t __make_mantissa(const char *__tagp) {
  if (!__tagp)
    return 0u;

  if (*__tagp == '0') {
    ++__tagp;

    if (*__tagp == 'x' || *__tagp == 'X')
      return __make_mantissa_base16(__tagp);
    else
      return __make_mantissa_base8(__tagp);
  }

  return __make_mantissa_base10(__tagp);
}
# 180 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/__clang_hip_math.h" 3
static __attribute__((always_inline, nothrow))
float acosf(float __x) { return __ocml_acos_f32(__x); }

static __attribute__((always_inline, nothrow))
float acoshf(float __x) { return __ocml_acosh_f32(__x); }

static __attribute__((always_inline, nothrow))
float asinf(float __x) { return __ocml_asin_f32(__x); }

static __attribute__((always_inline, nothrow))
float asinhf(float __x) { return __ocml_asinh_f32(__x); }

static __attribute__((always_inline, nothrow))
float atan2f(float __x, float __y) { return __ocml_atan2_f32(__x, __y); }

static __attribute__((always_inline, nothrow))
float atanf(float __x) { return __ocml_atan_f32(__x); }

static __attribute__((always_inline, nothrow))
float atanhf(float __x) { return __ocml_atanh_f32(__x); }

static __attribute__((always_inline, nothrow))
float cbrtf(float __x) { return __ocml_cbrt_f32(__x); }

static __attribute__((always_inline, nothrow))
float ceilf(float __x) { return __ocml_ceil_f32(__x); }

static __attribute__((always_inline, nothrow))
float copysignf(float __x, float __y) { return __ocml_copysign_f32(__x, __y); }

static __attribute__((always_inline, nothrow))
float cosf(float __x) { return __ocml_cos_f32(__x); }

static __attribute__((always_inline, nothrow))
float coshf(float __x) { return __ocml_cosh_f32(__x); }

static __attribute__((always_inline, nothrow))
float cospif(float __x) { return __ocml_cospi_f32(__x); }

static __attribute__((always_inline, nothrow))
float cyl_bessel_i0f(float __x) { return __ocml_i0_f32(__x); }

static __attribute__((always_inline, nothrow))
float cyl_bessel_i1f(float __x) { return __ocml_i1_f32(__x); }

static __attribute__((always_inline, nothrow))
float erfcf(float __x) { return __ocml_erfc_f32(__x); }

static __attribute__((always_inline, nothrow))
float erfcinvf(float __x) { return __ocml_erfcinv_f32(__x); }

static __attribute__((always_inline, nothrow))
float erfcxf(float __x) { return __ocml_erfcx_f32(__x); }

static __attribute__((always_inline, nothrow))
float erff(float __x) { return __ocml_erf_f32(__x); }

static __attribute__((always_inline, nothrow))
float erfinvf(float __x) { return __ocml_erfinv_f32(__x); }

static __attribute__((always_inline, nothrow))
float exp10f(float __x) { return __ocml_exp10_f32(__x); }

static __attribute__((always_inline, nothrow))
float exp2f(float __x) { return __ocml_exp2_f32(__x); }

static __attribute__((always_inline, nothrow))
float expf(float __x) { return __ocml_exp_f32(__x); }

static __attribute__((always_inline, nothrow))
float expm1f(float __x) { return __ocml_expm1_f32(__x); }

static __attribute__((always_inline, nothrow))
float fabsf(float __x) { return __ocml_fabs_f32(__x); }

static __attribute__((always_inline, nothrow))
float fdimf(float __x, float __y) { return __ocml_fdim_f32(__x, __y); }

static __attribute__((always_inline, nothrow))
float fdividef(float __x, float __y) { return __x / __y; }

static __attribute__((always_inline, nothrow))
float floorf(float __x) { return __ocml_floor_f32(__x); }

static __attribute__((always_inline, nothrow))
float fmaf(float __x, float __y, float __z) {
  return __ocml_fma_f32(__x, __y, __z);
}

static __attribute__((always_inline, nothrow))
float fmaxf(float __x, float __y) { return __ocml_fmax_f32(__x, __y); }

static __attribute__((always_inline, nothrow))
float fminf(float __x, float __y) { return __ocml_fmin_f32(__x, __y); }

static __attribute__((always_inline, nothrow))
float fmodf(float __x, float __y) { return __ocml_fmod_f32(__x, __y); }

static __attribute__((always_inline, nothrow))
float frexpf(float __x, int *__nptr) {

  static __attribute__((address_space(5))) int __tmp;
  float __r = __ocml_frexp_f32(__x, &__tmp);





  *__nptr = __tmp;

  return __r;
}

static __attribute__((always_inline, nothrow))
float hypotf(float __x, float __y) { return __ocml_hypot_f32(__x, __y); }

static __attribute__((always_inline, nothrow))
int ilogbf(float __x) { return __ocml_ilogb_f32(__x); }

static __attribute__((always_inline, nothrow))
int __finitef(float __x) { return __ocml_isfinite_f32(__x); }

static __attribute__((always_inline, nothrow))
int __isinff(float __x) { return __ocml_isinf_f32(__x); }

static __attribute__((always_inline, nothrow))
int __isnanf(float __x) { return __ocml_isnan_f32(__x); }

static __attribute__((always_inline, nothrow))
float j0f(float __x) { return __ocml_j0_f32(__x); }

static __attribute__((always_inline, nothrow))
float j1f(float __x) { return __ocml_j1_f32(__x); }

static __attribute__((always_inline, nothrow))
float jnf(int __n, float __x) {



  if (__n == 0)
    return j0f(__x);
  if (__n == 1)
    return j1f(__x);

  float __x0 = j0f(__x);
  float __x1 = j1f(__x);
  for (int __i = 1; __i < __n; ++__i) {
    float __x2 = (2 * __i) / __x * __x1 - __x0;
    __x0 = __x1;
    __x1 = __x2;
  }

  return __x1;
}

static __attribute__((always_inline, nothrow))
float ldexpf(float __x, int __e) { return __ocml_ldexp_f32(__x, __e); }

static __attribute__((always_inline, nothrow))
float lgammaf(float __x) { return __ocml_lgamma_f32(__x); }

static __attribute__((always_inline, nothrow))
long long int llrintf(float __x) { return __ocml_rint_f32(__x); }

static __attribute__((always_inline, nothrow))
long long int llroundf(float __x) { return __ocml_round_f32(__x); }

static __attribute__((always_inline, nothrow))
float log10f(float __x) { return __ocml_log10_f32(__x); }

static __attribute__((always_inline, nothrow))
float log1pf(float __x) { return __ocml_log1p_f32(__x); }

static __attribute__((always_inline, nothrow))
float log2f(float __x) { return __ocml_log2_f32(__x); }

static __attribute__((always_inline, nothrow))
float log2fi(int __x) { return __ocml_log2_f32((float) __x); }

static __attribute__((always_inline, nothrow))
float logbf(float __x) { return __ocml_logb_f32(__x); }

static __attribute__((always_inline, nothrow))
float logf(float __x) { return __ocml_log_f32(__x); }

static __attribute__((always_inline, nothrow))
long int lrintf(float __x) { return __ocml_rint_f32(__x); }

static __attribute__((always_inline, nothrow))
long int lroundf(float __x) { return __ocml_round_f32(__x); }

static __attribute__((always_inline, nothrow))
float modff(float __x, float *__iptr) {

  static __attribute__((address_space(5))) float __tmp;
  float __r = __ocml_modf_f32(__x, &__tmp);





  *__iptr = __tmp;
  return __r;
}
# 409 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/__clang_hip_math.h" 3
static __attribute__((always_inline, nothrow))
float nearbyintf(float __x) { return __ocml_nearbyint_f32(__x); }

static __attribute__((always_inline, nothrow))
float nextafterf(float __x, float __y) {
  return __ocml_nextafter_f32(__x, __y);
}

static __attribute__((always_inline, nothrow))
float norm3df(float __x, float __y, float __z) {
  return __ocml_len3_f32(__x, __y, __z);
}

static __attribute__((always_inline, nothrow))
float norm4df(float __x, float __y, float __z, float __w) {
  return __ocml_len4_f32(__x, __y, __z, __w);
}

static __attribute__((always_inline, nothrow))
float normcdff(float __x) { return __ocml_ncdf_f32(__x); }

static __attribute__((always_inline, nothrow))
float normcdfinvf(float __x) { return __ocml_ncdfinv_f32(__x); }

static __attribute__((always_inline, nothrow))
float normf(int __dim,
            const float *__a) {
  float __r = 0;
  while (__dim--) {
    __r += __a[0] * __a[0];
    ++__a;
  }

  return __ocml_sqrt_f32(__r);
}

static __attribute__((always_inline, nothrow))
float powf(float __x, float __y) { return __ocml_pow_f32(__x, __y); }

static __attribute__((always_inline, nothrow))
float powif(float __x, int __y) { return __ocml_pown_f32(__x, __y); }

static __attribute__((always_inline, nothrow))
int powii(int __base, int __exp) {
  if (__exp < 0 )
    return -1;
  int __result = 1;
  for (;;) {
    if (__exp & 1)
      __result *= __base;
    __exp >>= 1;
    if (!__exp)
      break;
    __base *= __base;
  }
  return __result;
}

static __attribute__((always_inline, nothrow))
float rcbrtf(float __x) { return __ocml_rcbrt_f32(__x); }

static __attribute__((always_inline, nothrow))
float remainderf(float __x, float __y) {
  return __ocml_remainder_f32(__x, __y);
}

static __attribute__((always_inline, nothrow))
float remquof(float __x, float __y, int *__quo) {

  static __attribute__((address_space(5))) int __tmp;
  float __r = __ocml_remquo_f32( __x, __y, &__tmp);





  *__quo = __tmp;

  return __r;
}

static __attribute__((always_inline, nothrow))
float rhypotf(float __x, float __y) { return __ocml_rhypot_f32(__x, __y); }

static __attribute__((always_inline, nothrow))
float rintf(float __x) { return __ocml_rint_f32(__x); }

static __attribute__((always_inline, nothrow))
float rnorm3df(float __x, float __y, float __z) {
  return __ocml_rlen3_f32(__x, __y, __z);
}

static __attribute__((always_inline, nothrow))
float rnorm4df(float __x, float __y, float __z, float __w) {
  return __ocml_rlen4_f32(__x, __y, __z, __w);
}

static __attribute__((always_inline, nothrow))
float rnormf(int __dim,
             const float *__a) {
  float __r = 0;
  while (__dim--) {
    __r += __a[0] * __a[0];
    ++__a;
  }

  return __ocml_rsqrt_f32(__r);
}

static __attribute__((always_inline, nothrow))
float roundf(float __x) { return __ocml_round_f32(__x); }

static __attribute__((always_inline, nothrow))
float rsqrtf(float __x) { return __ocml_rsqrt_f32(__x); }

static __attribute__((always_inline, nothrow))
float scalblnf(float __x, long int __n) {
  return (__n < 2147483647) ? __ocml_scalbn_f32(__x, __n)
                         : __ocml_scalb_f32(__x, __n);
}

static __attribute__((always_inline, nothrow))
float scalbnf(float __x, int __n) { return __ocml_scalbn_f32(__x, __n); }

static __attribute__((always_inline, nothrow))
int __signbitf(float __x) { return __ocml_signbit_f32(__x); }

static __attribute__((always_inline, nothrow))
void sincosf(float __x, float *__sinptr, float *__cosptr) {

  static __attribute__((address_space(5))) float __tmp;
  *__sinptr = __ocml_sincos_f32(__x, &__tmp);





  *__cosptr = __tmp;
}

static __attribute__((always_inline, nothrow))
void sincospif(float __x, float *__sinptr, float *__cosptr) {

  static __attribute__((address_space(5))) float __tmp;
  *__sinptr = __ocml_sincospi_f32(__x, &__tmp);





  *__cosptr = __tmp;
}

static __attribute__((always_inline, nothrow))
float sinf(float __x) { return __ocml_sin_f32(__x); }

static __attribute__((always_inline, nothrow))
float sinhf(float __x) { return __ocml_sinh_f32(__x); }

static __attribute__((always_inline, nothrow))
float sinpif(float __x) { return __ocml_sinpi_f32(__x); }

static __attribute__((always_inline, nothrow))
float sqrtf(float __x) { return __ocml_sqrt_f32(__x); }

static __attribute__((always_inline, nothrow))
float tanf(float __x) { return __ocml_tan_f32(__x); }

static __attribute__((always_inline, nothrow))
float tanhf(float __x) { return __ocml_tanh_f32(__x); }

static __attribute__((always_inline, nothrow))
float tgammaf(float __x) { return __ocml_tgamma_f32(__x); }

static __attribute__((always_inline, nothrow))
float truncf(float __x) { return __ocml_trunc_f32(__x); }

static __attribute__((always_inline, nothrow))
float y0f(float __x) { return __ocml_y0_f32(__x); }

static __attribute__((always_inline, nothrow))
float y1f(float __x) { return __ocml_y1_f32(__x); }

static __attribute__((always_inline, nothrow))
float ynf(int __n, float __x) {




  if (__n == 0)
    return y0f(__x);
  if (__n == 1)
    return y1f(__x);

  float __x0 = y0f(__x);
  float __x1 = y1f(__x);
  for (int __i = 1; __i < __n; ++__i) {
    float __x2 = (2 * __i) / __x * __x1 - __x0;
    __x0 = __x1;
    __x1 = __x2;
  }

  return __x1;
}



static __attribute__((always_inline, nothrow))
float __cosf(float __x) { return __ocml_native_cos_f32(__x); }

static __attribute__((always_inline, nothrow))
float __exp10f(float __x) { return __ocml_native_exp10_f32(__x); }

static __attribute__((always_inline, nothrow))
float __expf(float __x) { return __ocml_native_exp_f32(__x); }
# 635 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/__clang_hip_math.h" 3
static __attribute__((always_inline, nothrow))
float __fadd_rn(float __x, float __y) { return __x + __y; }
# 649 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/__clang_hip_math.h" 3
static __attribute__((always_inline, nothrow))
float __fdiv_rn(float __x, float __y) { return __x / __y; }


static __attribute__((always_inline, nothrow))
float __fdividef(float __x, float __y) { return __x / __y; }
# 674 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/__clang_hip_math.h" 3
static __attribute__((always_inline, nothrow))
float __fmaf_rn(float __x, float __y, float __z) {
  return __ocml_fma_f32(__x, __y, __z);
}
# 690 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/__clang_hip_math.h" 3
static __attribute__((always_inline, nothrow))
float __fmul_rn(float __x, float __y) { return __x * __y; }
# 704 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/__clang_hip_math.h" 3
static __attribute__((always_inline, nothrow))
float __frcp_rn(float __x) { return 1.0f / __x; }


static __attribute__((always_inline, nothrow))
float __frsqrt_rn(float __x) { return __llvm_amdgcn_rsq_f32(__x); }
# 721 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/__clang_hip_math.h" 3
static __attribute__((always_inline, nothrow))
float __fsqrt_rn(float __x) { return __ocml_native_sqrt_f32(__x); }
# 735 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/__clang_hip_math.h" 3
static __attribute__((always_inline, nothrow))
float __fsub_rn(float __x, float __y) { return __x - __y; }


static __attribute__((always_inline, nothrow))
float __log10f(float __x) { return __ocml_native_log10_f32(__x); }

static __attribute__((always_inline, nothrow))
float __log2f(float __x) { return __ocml_native_log2_f32(__x); }

static __attribute__((always_inline, nothrow))
float __logf(float __x) { return __ocml_native_log_f32(__x); }

static __attribute__((always_inline, nothrow))
float __powf(float __x, float __y) { return __ocml_pow_f32(__x, __y); }

static __attribute__((always_inline, nothrow))
float __saturatef(float __x) { return (__x < 0) ? 0 : ((__x > 1) ? 1 : __x); }

static __attribute__((always_inline, nothrow))
void __sincosf(float __x, float *__sinptr, float *__cosptr) {
  *__sinptr = __ocml_native_sin_f32(__x);
  *__cosptr = __ocml_native_cos_f32(__x);
}

static __attribute__((always_inline, nothrow))
float __sinf(float __x) { return __ocml_native_sin_f32(__x); }

static __attribute__((always_inline, nothrow))
float __tanf(float __x) { return __ocml_tan_f32(__x); }




static __attribute__((always_inline, nothrow))
double acos(double __x) { return __ocml_acos_f64(__x); }

static __attribute__((always_inline, nothrow))
double acosh(double __x) { return __ocml_acosh_f64(__x); }

static __attribute__((always_inline, nothrow))
double asin(double __x) { return __ocml_asin_f64(__x); }

static __attribute__((always_inline, nothrow))
double asinh(double __x) { return __ocml_asinh_f64(__x); }

static __attribute__((always_inline, nothrow))
double atan(double __x) { return __ocml_atan_f64(__x); }

static __attribute__((always_inline, nothrow))
double atan2(double __x, double __y) { return __ocml_atan2_f64(__x, __y); }

static __attribute__((always_inline, nothrow))
double atanh(double __x) { return __ocml_atanh_f64(__x); }

static __attribute__((always_inline, nothrow))
double cbrt(double __x) { return __ocml_cbrt_f64(__x); }

static __attribute__((always_inline, nothrow))
double ceil(double __x) { return __ocml_ceil_f64(__x); }

static __attribute__((always_inline, nothrow))
double copysign(double __x, double __y) {
  return __ocml_copysign_f64(__x, __y);
}

static __attribute__((always_inline, nothrow))
double cos(double __x) { return __ocml_cos_f64(__x); }

static __attribute__((always_inline, nothrow))
double cosh(double __x) { return __ocml_cosh_f64(__x); }

static __attribute__((always_inline, nothrow))
double cospi(double __x) { return __ocml_cospi_f64(__x); }

static __attribute__((always_inline, nothrow))
double cyl_bessel_i0(double __x) { return __ocml_i0_f64(__x); }

static __attribute__((always_inline, nothrow))
double cyl_bessel_i1(double __x) { return __ocml_i1_f64(__x); }

static __attribute__((always_inline, nothrow))
double erf(double __x) { return __ocml_erf_f64(__x); }

static __attribute__((always_inline, nothrow))
double erfc(double __x) { return __ocml_erfc_f64(__x); }

static __attribute__((always_inline, nothrow))
double erfcinv(double __x) { return __ocml_erfcinv_f64(__x); }

static __attribute__((always_inline, nothrow))
double erfcx(double __x) { return __ocml_erfcx_f64(__x); }

static __attribute__((always_inline, nothrow))
double erfinv(double __x) { return __ocml_erfinv_f64(__x); }

static __attribute__((always_inline, nothrow))
double exp(double __x) { return __ocml_exp_f64(__x); }

static __attribute__((always_inline, nothrow))
double exp10(double __x) { return __ocml_exp10_f64(__x); }

static __attribute__((always_inline, nothrow))
double exp2(double __x) { return __ocml_exp2_f64(__x); }

static __attribute__((always_inline, nothrow))
double expm1(double __x) { return __ocml_expm1_f64(__x); }

static __attribute__((always_inline, nothrow))
double fabs(double __x) { return __ocml_fabs_f64(__x); }

static __attribute__((always_inline, nothrow))
double fdim(double __x, double __y) { return __ocml_fdim_f64(__x, __y); }

static __attribute__((always_inline, nothrow))
double floor(double __x) { return __ocml_floor_f64(__x); }

static __attribute__((always_inline, nothrow))
double fma(double __x, double __y, double __z) {
  return __ocml_fma_f64(__x, __y, __z);
}

static __attribute__((always_inline, nothrow))
double fmax(double __x, double __y) { return __ocml_fmax_f64(__x, __y); }

static __attribute__((always_inline, nothrow))
double fmin(double __x, double __y) { return __ocml_fmin_f64(__x, __y); }

static __attribute__((always_inline, nothrow))
double fmod(double __x, double __y) { return __ocml_fmod_f64(__x, __y); }

static __attribute__((always_inline, nothrow))
double frexp(double __x, int *__nptr) {

  static __attribute__((address_space(5))) int __tmp;
  double __r = __ocml_frexp_f64(__x, &__tmp);





  *__nptr = __tmp;
  return __r;
}

static __attribute__((always_inline, nothrow))
double hypot(double __x, double __y) { return __ocml_hypot_f64(__x, __y); }

static __attribute__((always_inline, nothrow))
int ilogb(double __x) { return __ocml_ilogb_f64(__x); }

static __attribute__((always_inline, nothrow))
int __finite(double __x) { return __ocml_isfinite_f64(__x); }

static __attribute__((always_inline, nothrow))
int __isinf(double __x) { return __ocml_isinf_f64(__x); }

static __attribute__((always_inline, nothrow))
int __isnan(double __x) { return __ocml_isnan_f64(__x); }

static __attribute__((always_inline, nothrow))
double j0(double __x) { return __ocml_j0_f64(__x); }

static __attribute__((always_inline, nothrow))
double j1(double __x) { return __ocml_j1_f64(__x); }

static __attribute__((always_inline, nothrow))
double jn(int __n, double __x) {




  if (__n == 0)
    return j0(__x);
  if (__n == 1)
    return j1(__x);

  double __x0 = j0(__x);
  double __x1 = j1(__x);
  for (int __i = 1; __i < __n; ++__i) {
    double __x2 = (2 * __i) / __x * __x1 - __x0;
    __x0 = __x1;
    __x1 = __x2;
  }
  return __x1;
}

static __attribute__((always_inline, nothrow))
double ldexp(double __x, int __e) { return __ocml_ldexp_f64(__x, __e); }

static __attribute__((always_inline, nothrow))
double lgamma(double __x) { return __ocml_lgamma_f64(__x); }

static __attribute__((always_inline, nothrow))
long long int llrint(double __x) { return __ocml_rint_f64(__x); }

static __attribute__((always_inline, nothrow))
long long int llround(double __x) { return __ocml_round_f64(__x); }

static __attribute__((always_inline, nothrow))
double log(double __x) { return __ocml_log_f64(__x); }

static __attribute__((always_inline, nothrow))
double log10(double __x) { return __ocml_log10_f64(__x); }

static __attribute__((always_inline, nothrow))
double log1p(double __x) { return __ocml_log1p_f64(__x); }

static __attribute__((always_inline, nothrow))
double log2(double __x) { return __ocml_log2_f64(__x); }

static __attribute__((always_inline, nothrow))
double logb(double __x) { return __ocml_logb_f64(__x); }

static __attribute__((always_inline, nothrow))
long int lrint(double __x) { return __ocml_rint_f64(__x); }

static __attribute__((always_inline, nothrow))
long int lround(double __x) { return __ocml_round_f64(__x); }

static __attribute__((always_inline, nothrow))
double modf(double __x, double *__iptr) {

  static __attribute__((address_space(5))) double __tmp;
  double __r = __ocml_modf_f64(__x, &__tmp);





  *__iptr = __tmp;

  return __r;
}
# 1001 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/__clang_hip_math.h" 3
static __attribute__((always_inline, nothrow))
double nearbyint(double __x) { return __ocml_nearbyint_f64(__x); }

static __attribute__((always_inline, nothrow))
double nextafter(double __x, double __y) {
  return __ocml_nextafter_f64(__x, __y);
}

static __attribute__((always_inline, nothrow))
double norm(int __dim,
            const double *__a) {
  double __r = 0;
  while (__dim--) {
    __r += __a[0] * __a[0];
    ++__a;
  }

  return __ocml_sqrt_f64(__r);
}

static __attribute__((always_inline, nothrow))
double norm3d(double __x, double __y, double __z) {
  return __ocml_len3_f64(__x, __y, __z);
}

static __attribute__((always_inline, nothrow))
double norm4d(double __x, double __y, double __z, double __w) {
  return __ocml_len4_f64(__x, __y, __z, __w);
}

static __attribute__((always_inline, nothrow))
double normcdf(double __x) { return __ocml_ncdf_f64(__x); }

static __attribute__((always_inline, nothrow))
double normcdfinv(double __x) { return __ocml_ncdfinv_f64(__x); }

static __attribute__((always_inline, nothrow))
double pow(double __x, double __y) { return __ocml_pow_f64(__x, __y); }

static __attribute__((always_inline, nothrow))
double powi(double __x, int __y) { return __ocml_pown_f64(__x, __y); }

static __attribute__((always_inline, nothrow))
double rcbrt(double __x) { return __ocml_rcbrt_f64(__x); }

static __attribute__((always_inline, nothrow))
double remainder(double __x, double __y) {
  return __ocml_remainder_f64(__x, __y);
}

static __attribute__((always_inline, nothrow))
double remquo(double __x, double __y, int *__quo) {

  static __attribute__((address_space(5))) int __tmp;
  double __r = __ocml_remquo_f64(__x, __y, &__tmp);





  *__quo = __tmp;

  return __r;
}

static __attribute__((always_inline, nothrow))
double rhypot(double __x, double __y) { return __ocml_rhypot_f64(__x, __y); }

static __attribute__((always_inline, nothrow))
double rint(double __x) { return __ocml_rint_f64(__x); }

static __attribute__((always_inline, nothrow))
double rnorm(int __dim,
             const double *__a) {
  double __r = 0;
  while (__dim--) {
    __r += __a[0] * __a[0];
    ++__a;
  }

  return __ocml_rsqrt_f64(__r);
}

static __attribute__((always_inline, nothrow))
double rnorm3d(double __x, double __y, double __z) {
  return __ocml_rlen3_f64(__x, __y, __z);
}

static __attribute__((always_inline, nothrow))
double rnorm4d(double __x, double __y, double __z, double __w) {
  return __ocml_rlen4_f64(__x, __y, __z, __w);
}

static __attribute__((always_inline, nothrow))
double round(double __x) { return __ocml_round_f64(__x); }

static __attribute__((always_inline, nothrow))
double rsqrt(double __x) { return __ocml_rsqrt_f64(__x); }

static __attribute__((always_inline, nothrow))
double scalbln(double __x, long int __n) {
  return (__n < 2147483647) ? __ocml_scalbn_f64(__x, __n)
                         : __ocml_scalb_f64(__x, __n);
}
static __attribute__((always_inline, nothrow))
double scalbn(double __x, int __n) { return __ocml_scalbn_f64(__x, __n); }

static __attribute__((always_inline, nothrow))
int __signbit(double __x) { return __ocml_signbit_f64(__x); }

static __attribute__((always_inline, nothrow))
double sin(double __x) { return __ocml_sin_f64(__x); }

static __attribute__((always_inline, nothrow))
void sincos(double __x, double *__sinptr, double *__cosptr) {

  static __attribute__((address_space(5))) double __tmp;
  *__sinptr = __ocml_sincos_f64(__x, &__tmp);





  *__cosptr = __tmp;
}

static __attribute__((always_inline, nothrow))
void sincospi(double __x, double *__sinptr, double *__cosptr) {

  static __attribute__((address_space(5))) double __tmp;
  *__sinptr = __ocml_sincospi_f64(__x, &__tmp);





  *__cosptr = __tmp;
}

static __attribute__((always_inline, nothrow))
double sinh(double __x) { return __ocml_sinh_f64(__x); }

static __attribute__((always_inline, nothrow))
double sinpi(double __x) { return __ocml_sinpi_f64(__x); }

static __attribute__((always_inline, nothrow))
double sqrt(double __x) { return __ocml_sqrt_f64(__x); }

static __attribute__((always_inline, nothrow))
double tan(double __x) { return __ocml_tan_f64(__x); }

static __attribute__((always_inline, nothrow))
double tanh(double __x) { return __ocml_tanh_f64(__x); }

static __attribute__((always_inline, nothrow))
double tgamma(double __x) { return __ocml_tgamma_f64(__x); }

static __attribute__((always_inline, nothrow))
double trunc(double __x) { return __ocml_trunc_f64(__x); }

static __attribute__((always_inline, nothrow))
double y0(double __x) { return __ocml_y0_f64(__x); }

static __attribute__((always_inline, nothrow))
double y1(double __x) { return __ocml_y1_f64(__x); }

static __attribute__((always_inline, nothrow))
double yn(int __n, double __x) {




  if (__n == 0)
    return y0(__x);
  if (__n == 1)
    return y1(__x);

  double __x0 = y0(__x);
  double __x1 = y1(__x);
  for (int __i = 1; __i < __n; ++__i) {
    double __x2 = (2 * __i) / __x * __x1 - __x0;
    __x0 = __x1;
    __x1 = __x2;
  }

  return __x1;
}
# 1208 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/__clang_hip_math.h" 3
static __attribute__((always_inline, nothrow))
double __dadd_rn(double __x, double __y) { return __x + __y; }
# 1230 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/__clang_hip_math.h" 3
static __attribute__((always_inline, nothrow))
double __ddiv_rn(double __x, double __y) { return __x / __y; }
# 1252 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/__clang_hip_math.h" 3
static __attribute__((always_inline, nothrow))
double __dmul_rn(double __x, double __y) { return __x * __y; }
# 1266 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/__clang_hip_math.h" 3
static __attribute__((always_inline, nothrow))
double __drcp_rn(double __x) { return 1.0 / __x; }
# 1280 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/__clang_hip_math.h" 3
static __attribute__((always_inline, nothrow))
double __dsqrt_rn(double __x) { return __ocml_sqrt_f64(__x); }
# 1302 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/__clang_hip_math.h" 3
static __attribute__((always_inline, nothrow))
double __dsub_rn(double __x, double __y) { return __x - __y; }
# 1324 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/__clang_hip_math.h" 3
static __attribute__((always_inline, nothrow))
double __fma_rn(double __x, double __y, double __z) {
  return __ocml_fma_f64(__x, __y, __z);
}
# 67 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/openmp_wrappers/math.h" 2 3
#pragma omp end declare variant
# 3 "print_results.c" 2
# 1 "../common/type.h" 1



typedef enum { false, true } logical;
typedef struct {
  double real;
  double imag;
} dcomplex;
# 4 "print_results.c" 2


void print_results(char *name, char class, int n1, int n2, int n3, int niter,
    double t, double mops, char *optype, logical verified, char *npbversion,
    char *compiletime, char *cs1, char *cs2, char *cs3, char *cs4, char *cs5,
    char *cs6, char *cs7)
{
  char size[16];
  int j;

  printf( "\n\n %s Benchmark Completed.\n", name );
  printf( " Class           =             %12c\n", class );






  if ( ( n2 == 0 ) && ( n3 == 0 ) ) {
    if ( ( name[0] == 'E' ) && ( name[1] == 'P' ) ) {
      sprintf( size, "%15.0lf", pow(2.0, n1) );
      j = 14;
      if ( size[j] == '.' ) {
        size[j] = ' ';
        j--;
      }
      size[j+1] = '\0';
      printf( " Size            =          %15s\n", size );
    } else {
      printf( " Size            =             %12d\n", n1 );
    }
  } else {
    printf( " Size            =           %4dx%4dx%4d\n", n1, n2, n3 );
  }

  printf( " Iterations      =             %12d\n", niter );
  printf( " Time in seconds =             %12.2lf\n", t );
  printf( " Mop/s total     =          %15.2lf\n", mops );
  printf( " Operation type  = %24s\n", optype );
  if ( verified )
    printf( " Verification    =             %12s\n", "SUCCESSFUL" );
  else
    printf( " Verification    =             %12s\n", "UNSUCCESSFUL" );
  printf( " Version         =             %12s\n", npbversion );
  printf( " Compile date    =             %12s\n", compiletime );

  printf( "\n Compile options:\n"
          "    CC           = %s\n", cs1 );
  printf( "    CLINK        = %s\n", cs2 );
  printf( "    C_LIB        = %s\n", cs3 );
  printf( "    C_INC        = %s\n", cs4 );
  printf( "    CFLAGS       = %s\n", cs5 );
  printf( "    CLINKFLAGS   = %s\n", cs6 );
  printf( "    RAND         = %s\n", cs7 );

  printf( "\n--------------------------------------\n"
          " Please send all errors/feedbacks to:\n"
          " Center for Manycore Programming\n"
          " cmp@aces.snu.ac.kr\n"
          " http://aces.snu.ac.kr\n"
          "--------------------------------------\n\n");
}
