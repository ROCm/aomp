**Heap Buffer Overflow** Demonstration on AMDGPU with applications written in HIP/OpenMP.
========================================================================================================================================

> ## **HIP** ##
>
> ```CPP
>
> __global__ void vecAdd(int *A,int *B,int *C,int N){
> int i = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;
> if(i < N){
>   C[i+10] = A[i] + B[i];
> }
> }
>
>```

>
> ## **OpenMP** ##
>
>
>```CPP
>
>#pragma omp target parallel for
>for(int i = 0; i < N; i++){
>    C[i+10] = A[i] + B[i];
>}
>
>=================================================================
>==3464643==ERROR: AddressSanitizer: >heap-buffer-overflow on amdgpu device 0 at pc 0x7f0d326189d8
>WRITE of size 4 in workgroup id (0,0,0)
>  #0 0x7f0d326189d8 in __omp_outlined__ at /home/>ampandey/device-asan/openmp/vecadd-HBO.cpp:31:8
>
>Thread ids and accessed addresses:
>90 : 0x7f0cdc609190 91 : 0x7f0cdc609194 92 : 0x7f0cdc609198 93 : 0x7f0cdc60919c 94 : 0x7f0cdc6091a0 95 : 0x7f0cdc6091a4 96 : 0x7f0cdc6091a8 97 : 0x7f0cdc6091ac
>98 : 0x7f0cdc6091b0 99 : 0x7f0cdc6091b4
>
>0x7f0cdc609190 is located 0 bytes to the right of 400-byte region [0x7f0cdc609000,0x7f0cdc609190)
>allocated by thread T0 here:
>    #0 0x7f0eae92afd8 in hsa_amd_memory_pool_allocate />home/ampandey/rocm-toolchain/staging/src-home/>llvm-project/compiler-rt/lib/asan/asan_interceptors.>cpp:622:3
>    #1 0x7f0ea065d67c in device_malloc /home/ampandey/rocm-toolchain/staging/src-home/llvm-project/openmp/libomptarget/plugins/amdgpu/src/rtl.cpp:3517:38
>    #2 0x7f0ea0666172 in RTLDeviceInfoTy::AMDGPUDeviceAllocatorTy::allocate(unsigned long, void*, TargetAllocTy) /home/ampandey/rocm-toolchain/staging/src-home/llvm-project/openmp/libomptarget/plugins/amdgpu/src/rtl.cpp:1087:41
>    #3 0x7f0ea065c9b9 in __tgt_rtl_data_alloc_impl /home/ampandey/rocm-toolchain/staging/src-home/llvm-project/openmp/libomptarget/plugins/amdgpu/src/rtl.cpp:3265:58
>    #4 0x7f0ea0652999 in __tgt_rtl_data_alloc /home/ampandey/rocm-toolchain/staging/src-home/llvm-project/openmp/libomptarget/plugins/amdgpu/src/trace.h:179:38
>    #5 0x7f0ea4d806cc in DeviceTy::allocData(long, void*, int) /home/ampandey/rocm-toolchain/staging/src-home/llvm-project/openmp/libomptarget/src/device.cpp:640:34
>    #6 0x7f0ea4d7ef9b in DeviceTy::getTargetPointer(void*, void*, long, void*, bool, bool, bool, bool, bool, bool, bool, AsyncInfoTy&) /home/ampandey/rocm-toolchain/staging/src-home/llvm-project/openmp/libomptarget/src/device.cpp:398:41
>    #7 0x7f0ea4db8a6b in targetDataBegin(ident_t*, DeviceTy&, int, void**, void**, long*, long*, void**, void**, AsyncInfoTy&, bool) /home/ampandey/rocm-toolchain/staging/src-home/llvm-project/openmp/libomptarget/src/omptarget.cpp:593:39
>    #8 0x7f0ea4d88394 in __tgt_target_data_begin_mapper /home/ampandey/rocm-toolchain/staging/src-home/llvm-project/openmp/libomptarget/src/interface.cpp:210:27
>    #9 0x56231fab4556 in main /home/ampandey/device-asan/openmp/vecadd-HBO.cpp:23:1
>    #10 0x7f0ea4b580b2 in __libc_start_main /build/glibc-sMfBJT/glibc-2.31/csu/../csu/libc-start.c:308:16
>
>Shadow bytes around the buggy address:
>  0x0fe21b8b91e0: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
>  0x0fe21b8b91f0: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
>  0x0fe21b8b9200: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
>  0x0fe21b8b9210: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
>  0x0fe21b8b9220: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
>=>0x0fe21b8b9230: 00 00[fa]fa fa fa fa fa fa fa fa fa fa fa fa fa
>  0x0fe21b8b9240: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
>  0x0fe21b8b9250: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
>  0x0fe21b8b9260: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
>  0x0fe21b8b9270: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
>  0x0fe21b8b9280: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
>Shadow byte legend (one shadow byte represents 8 application bytes):
>  Addressable:           00
>  Partially addressable: 01 02 03 04 05 06 07
>  Heap left redzone:       fa
>  Freed heap region:       fd
>  Stack left redzone:      f1
>  Stack mid redzone:       f2
>  Stack right redzone:     f3
>  Stack after return:      f5
>  Stack use after scope:   f8
>  Global redzone:          f9
>  Global init order:       f6
>  Poisoned by user:        f7
>  Container overflow:      fc
>  Array cookie:            ac
>  Intra object redzone:    bb
>  ASan internal:           fe
>  Left alloca redzone:     ca
>  Right alloca redzone:    cb
>==3464643==ABORTING
>
>
```
