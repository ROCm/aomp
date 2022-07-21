**Global Buffer Overflow** Demonstration on AMDGPU with applications written in HIP/OpenMP.
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
>==3472068==ERROR: AddressSanitizer: global-buffer-overflow on amdgpu device 0 at pc 0x7f150ae13cb4
>WRITE of size 4 in workgroup id (0,0,0)
>  #0 0x7f150ae13cb4 in __omp_outlined__ at /home/ampandey/device-asan/openmp/vecadd-GBO.cpp:32:8
>
>Thread ids and accessed addresses:
>90 : 0x7f150ae7a850 91 : 0x7f150ae7a854 92 : 0x7f150ae7a858 93 : 0x7f150ae7a85c 94 : 0x7f150ae7a860 95 : 0x7f150ae7a864 96 : 0x7f150ae7a868 97 : 0x7f150ae7a86c
>98 : 0x7f150ae7a870
>
>Address 0x7f150ae7a850 is a wild pointer inside of access range of size 0x000000000004.
>Shadow bytes around the buggy address:
>   0x0fe3215c74b0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
>   0x0fe3215c74c0: 00 00 00 00 00 00 00 00 00 00 f9 f9 f9 f9 f9 f9
>   0x0fe3215c74d0: f9 f9 f9 f9 f9 f9 f9 f9 00 00 00 00 00 00 00 00
>   0x0fe3215c74e0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
>   0x0fe3215c74f0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
> =>0x0fe3215c7500: 00 00 00 00 00 00 00 00 00 00[f9]f9 f9 f9 f9 f9
>   0x0fe3215c7510: f9 f9 f9 f9 f9 f9 f9 f9 00 00 00 00 00 00 00 00
>   0x0fe3215c7520: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
>   0x0fe3215c7530: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
>   0x0fe3215c7540: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
>   0x0fe3215c7550: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
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
>==3472068==ABORTING
'''
>
