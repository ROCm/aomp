# 1 "../common/wtime.c"
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
# 1 "../common/wtime.c" 2
# 1 "../common/wtime.h" 1
# 2 "../common/wtime.c" 2
# 1 "/usr/include/time.h" 1 3 4
# 29 "/usr/include/time.h" 3 4
# 1 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/stddef.h" 1 3 4
# 46 "/opt/rocm-4.1.0/llvm/lib/clang/12.0.0/include/stddef.h" 3 4
typedef long unsigned int size_t;
# 30 "/usr/include/time.h" 2 3 4



# 1 "/usr/include/bits/time.h" 1 3 4
# 34 "/usr/include/time.h" 2 3 4



# 1 "/usr/include/bits/types/clock_t.h" 1 3 4






typedef __clock_t clock_t;
# 38 "/usr/include/time.h" 2 3 4
# 1 "/usr/include/bits/types/time_t.h" 1 3 4






typedef __time_t time_t;
# 39 "/usr/include/time.h" 2 3 4
# 1 "/usr/include/bits/types/struct_tm.h" 1 3 4






struct tm
{
  int tm_sec;
  int tm_min;
  int tm_hour;
  int tm_mday;
  int tm_mon;
  int tm_year;
  int tm_wday;
  int tm_yday;
  int tm_isdst;


  long int tm_gmtoff;
  const char *tm_zone;




};
# 40 "/usr/include/time.h" 2 3 4


# 1 "/usr/include/bits/types/struct_timespec.h" 1 3 4








struct timespec
{
  __time_t tv_sec;
  __syscall_slong_t tv_nsec;
};
# 43 "/usr/include/time.h" 2 3 4



# 1 "/usr/include/bits/types/clockid_t.h" 1 3 4






typedef __clockid_t clockid_t;
# 47 "/usr/include/time.h" 2 3 4
# 1 "/usr/include/bits/types/timer_t.h" 1 3 4






typedef __timer_t timer_t;
# 48 "/usr/include/time.h" 2 3 4
# 1 "/usr/include/bits/types/struct_itimerspec.h" 1 3 4







struct itimerspec
  {
    struct timespec it_interval;
    struct timespec it_value;
  };
# 49 "/usr/include/time.h" 2 3 4
struct sigevent;




typedef __pid_t pid_t;






# 1 "/usr/include/bits/types/locale_t.h" 1 3 4
# 22 "/usr/include/bits/types/locale_t.h" 3 4
# 1 "/usr/include/bits/types/__locale_t.h" 1 3 4
# 28 "/usr/include/bits/types/__locale_t.h" 3 4
struct __locale_struct
{

  struct __locale_data *__locales[13];


  const unsigned short int *__ctype_b;
  const int *__ctype_tolower;
  const int *__ctype_toupper;


  const char *__names[13];
};

typedef struct __locale_struct *__locale_t;
# 23 "/usr/include/bits/types/locale_t.h" 2 3 4

typedef __locale_t locale_t;
# 61 "/usr/include/time.h" 2 3 4
# 72 "/usr/include/time.h" 3 4
extern clock_t clock (void) __attribute__ ((__nothrow__ ));


extern time_t time (time_t *__timer) __attribute__ ((__nothrow__ ));


extern double difftime (time_t __time1, time_t __time0)
     __attribute__ ((__nothrow__ )) __attribute__ ((__const__));


extern time_t mktime (struct tm *__tp) __attribute__ ((__nothrow__ ));





extern size_t strftime (char *__restrict __s, size_t __maxsize,
   const char *__restrict __format,
   const struct tm *__restrict __tp) __attribute__ ((__nothrow__ ));
# 104 "/usr/include/time.h" 3 4
extern size_t strftime_l (char *__restrict __s, size_t __maxsize,
     const char *__restrict __format,
     const struct tm *__restrict __tp,
     locale_t __loc) __attribute__ ((__nothrow__ ));
# 119 "/usr/include/time.h" 3 4
extern struct tm *gmtime (const time_t *__timer) __attribute__ ((__nothrow__ ));



extern struct tm *localtime (const time_t *__timer) __attribute__ ((__nothrow__ ));




extern struct tm *gmtime_r (const time_t *__restrict __timer,
       struct tm *__restrict __tp) __attribute__ ((__nothrow__ ));



extern struct tm *localtime_r (const time_t *__restrict __timer,
          struct tm *__restrict __tp) __attribute__ ((__nothrow__ ));




extern char *asctime (const struct tm *__tp) __attribute__ ((__nothrow__ ));


extern char *ctime (const time_t *__timer) __attribute__ ((__nothrow__ ));






extern char *asctime_r (const struct tm *__restrict __tp,
   char *__restrict __buf) __attribute__ ((__nothrow__ ));


extern char *ctime_r (const time_t *__restrict __timer,
        char *__restrict __buf) __attribute__ ((__nothrow__ ));




extern char *__tzname[2];
extern int __daylight;
extern long int __timezone;




extern char *tzname[2];



extern void tzset (void) __attribute__ ((__nothrow__ ));



extern int daylight;
extern long int timezone;





extern int stime (const time_t *__when) __attribute__ ((__nothrow__ ));
# 196 "/usr/include/time.h" 3 4
extern time_t timegm (struct tm *__tp) __attribute__ ((__nothrow__ ));


extern time_t timelocal (struct tm *__tp) __attribute__ ((__nothrow__ ));


extern int dysize (int __year) __attribute__ ((__nothrow__ )) __attribute__ ((__const__));
# 211 "/usr/include/time.h" 3 4
extern int nanosleep (const struct timespec *__requested_time,
        struct timespec *__remaining);



extern int clock_getres (clockid_t __clock_id, struct timespec *__res) __attribute__ ((__nothrow__ ));


extern int clock_gettime (clockid_t __clock_id, struct timespec *__tp) __attribute__ ((__nothrow__ ));


extern int clock_settime (clockid_t __clock_id, const struct timespec *__tp)
     __attribute__ ((__nothrow__ ));






extern int clock_nanosleep (clockid_t __clock_id, int __flags,
       const struct timespec *__req,
       struct timespec *__rem);


extern int clock_getcpuclockid (pid_t __pid, clockid_t *__clock_id) __attribute__ ((__nothrow__ ));




extern int timer_create (clockid_t __clock_id,
    struct sigevent *__restrict __evp,
    timer_t *__restrict __timerid) __attribute__ ((__nothrow__ ));


extern int timer_delete (timer_t __timerid) __attribute__ ((__nothrow__ ));


extern int timer_settime (timer_t __timerid, int __flags,
     const struct itimerspec *__restrict __value,
     struct itimerspec *__restrict __ovalue) __attribute__ ((__nothrow__ ));


extern int timer_gettime (timer_t __timerid, struct itimerspec *__value)
     __attribute__ ((__nothrow__ ));


extern int timer_getoverrun (timer_t __timerid) __attribute__ ((__nothrow__ ));





extern int timespec_get (struct timespec *__ts, int __base)
     __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1)));
# 3 "../common/wtime.c" 2

# 1 "/usr/include/sys/time.h" 1 3 4
# 25 "/usr/include/sys/time.h" 3 4
# 1 "/usr/include/bits/types/struct_timeval.h" 1 3 4







struct timeval
{
  __time_t tv_sec;
  __suseconds_t tv_usec;
};
# 26 "/usr/include/sys/time.h" 2 3 4


typedef __suseconds_t suseconds_t;




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
# 49 "/usr/include/sys/select.h" 3 4
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
# 33 "/usr/include/sys/time.h" 2 3 4
# 52 "/usr/include/sys/time.h" 3 4
struct timezone
  {
    int tz_minuteswest;
    int tz_dsttime;
  };

typedef struct timezone *__restrict __timezone_ptr_t;
# 68 "/usr/include/sys/time.h" 3 4
extern int gettimeofday (struct timeval *__restrict __tv,
    __timezone_ptr_t __tz) __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1)));




extern int settimeofday (const struct timeval *__tv,
    const struct timezone *__tz)
     __attribute__ ((__nothrow__ ));





extern int adjtime (const struct timeval *__delta,
      struct timeval *__olddelta) __attribute__ ((__nothrow__ ));




enum __itimer_which
  {

    ITIMER_REAL = 0,


    ITIMER_VIRTUAL = 1,



    ITIMER_PROF = 2

  };



struct itimerval
  {

    struct timeval it_interval;

    struct timeval it_value;
  };






typedef int __itimer_which_t;




extern int getitimer (__itimer_which_t __which,
        struct itimerval *__value) __attribute__ ((__nothrow__ ));




extern int setitimer (__itimer_which_t __which,
        const struct itimerval *__restrict __new,
        struct itimerval *__restrict __old) __attribute__ ((__nothrow__ ));




extern int utimes (const char *__file, const struct timeval __tvp[2])
     __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1)));



extern int lutimes (const char *__file, const struct timeval __tvp[2])
     __attribute__ ((__nothrow__ )) __attribute__ ((__nonnull__ (1)));


extern int futimes (int __fd, const struct timeval __tvp[2]) __attribute__ ((__nothrow__ ));
# 5 "../common/wtime.c" 2


void wtime_(double *t)
{
  static int sec = -1;
  struct timeval tv;
  gettimeofday(&tv, (void *)0);
  if (sec < 0) sec = tv.tv_sec;
  *t = (tv.tv_sec - sec) + 1.0e-6*tv.tv_usec;
}
