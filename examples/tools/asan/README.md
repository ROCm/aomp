The AOMP compiler supports ASAN for AMDGPU with applications written in both HIP and OpenMP.
========================================================================================================================================

Requirements
========================================================================================================================================
ASAN for AMDGPU requires amdgpu with capability xnack+.

Steps to build ASAN for AMDGPU using LLVM COMPILER
========================================================================================================================================
1. ASAN Patches for AOMP compiler are upstreamed in amd-stg-open branch.
2. ASAN patches for AMDGPU at the stage of compiler-rt runtime and openmp runtime(libomptarget) are controlled by flag SANITIZER_AMDGPU.
3. Make sure to build the AOMP compiler with flag -DSANITIZER_AMDGPU=1.
