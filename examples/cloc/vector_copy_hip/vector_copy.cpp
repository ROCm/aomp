/* Copyright 2014 HSA Foundation Inc.  All Rights Reserved.
 *
 * HSAF is granting you permission to use this software and documentation (if
 * any) (collectively, the "Materials") pursuant to the terms and conditions
 * of the Software License Agreement included with the Materials.  If you do
 * not have a copy of the Software License Agreement, contact the  HSA Foundation for a copy.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE SOFTWARE.
 */

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <hip/hip_hcc.h>

//#define GLOBAL_SIZE 1024*1024
#define GLOBAL_SIZE 1024*1024
#define BLOCK_SIZE 512
#define GRID_SIZE GLOBAL_SIZE/BLOCK_SIZE

// Code object filename.
#define fileName "vector_copy.hsaco"
#define kernel_name "vector_copy"

#define HIP_CHECK(status)                                                                          \
    if (status != hipSuccess) {                                                                    \
        std::cout << "Got Status: " << status << " at Line: " << __LINE__ << std::endl;            \
        exit(0);                                                                                   \
    }

int main() {
    const size_t sizeBytes = GLOBAL_SIZE * sizeof(uint32_t);
    uint32_t *in, *out;
    hipDeviceptr_t in_d, out_d;
    in = new uint32_t[GLOBAL_SIZE];
    out = new uint32_t[GLOBAL_SIZE];

    for (uint32_t i = 0; i < GLOBAL_SIZE; i++) {
        in[i] = i;
        out[i] = 0xff;
    }

    hipInit(0);
    hipDevice_t device;
    hipCtx_t context;
    hipSetDevice(0);

    hipMalloc((void**)&in_d, sizeBytes);
    hipMalloc((void**)&out_d, sizeBytes);

    hipMemcpyHtoD(in_d, in, sizeBytes);
    hipMemcpyHtoD(out_d, out, sizeBytes);
    hipModule_t Module;
    hipFunction_t Function;
    HIP_CHECK(hipModuleLoad(&Module, fileName));
    HIP_CHECK(hipModuleGetFunction(&Function, Module, kernel_name));

#ifdef __HIP_PLATFORM_HCC__
    uint32_t len = GLOBAL_SIZE;
    uint32_t one = 1;

    struct {
        void* _in_d;
        void* _out_d;
    } args;

    args._in_d = in_d;
    args._out_d = out_d;

#endif

#ifdef __HIP_PLATFORM_NVCC__
    struct {
        uint32_t _hidden[1];
        void* _in_d;
        void* _out_d;
    } args;

    args._hidden[0] = 0;
    args._in_d = in_d;
    args._out_d = out_d;
#endif


    size_t size = sizeof(args);

    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                      HIP_LAUNCH_PARAM_END};

    HIP_CHECK(hipModuleLaunchKernel(Function, GRID_SIZE, 1, 1, BLOCK_SIZE, 1, 1, 0, 0, NULL, (void**)&config));

    hipMemcpyDtoH(out, out_d, sizeBytes);

    int mismatchCount = 0;
    for (uint32_t i = 0; i < GLOBAL_SIZE; i++) {
        if (in[i] != out[i]) {
            mismatchCount++;
            std::cout << "error: mismatch " << in[i] << " != " << out[i] << std::endl;
	    break;
        }
    }

    if (mismatchCount == 0) {
        std::cout << "PASSED!\n";
    } else {
        std::cout << "FAILED!\n";
    };

    return 0;
}
