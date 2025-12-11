#include "hip/hip_runtime.h"
#include <stdio.h>
#include <cmath>
#include <cstring>

#define HIP_CHECK(cmd) \
    do { \
        hipError_t error = cmd; \
        if (error != hipSuccess) { \
            fprintf(stderr, "HIP error: '%s'(%d) at %s:%d\n", \
                    hipGetErrorString(error), error, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__global__ void test_log(float* result, float input) {
    *result = ::log(input);
}
__global__ void test_log10(float* result, float input) {
    *result = ::log10(input);
}

// Calculate ULP error between two float values
float ulp_error(float gpu_result, float cpu_reference) {
    // Handle special cases
    if (std::isnan(gpu_result) && std::isnan(cpu_reference)) {
        return 0.0f;
    }
    if (std::isnan(gpu_result) || std::isnan(cpu_reference)) {
        return INFINITY;
    }
    if (std::isinf(gpu_result) && std::isinf(cpu_reference)) {
        if ((gpu_result > 0) == (cpu_reference > 0)) {
            return 0.0f;
        } else {
            return INFINITY;
        }
    }
    
    uint32_t gpu_bits, cpu_bits;
    std::memcpy(&gpu_bits, &gpu_result, sizeof(float));
    std::memcpy(&cpu_bits, &cpu_reference, sizeof(float));
    
    // Handle sign bit differences
    if ((gpu_bits ^ cpu_bits) & 0x80000000) {
        // Different signs - treat zero cases specially
        if (gpu_bits == 0x80000000 && cpu_bits == 0) return 0.0f; // -0.0 vs +0.0
        if (gpu_bits == 0 && cpu_bits == 0x80000000) return 0.0f; // +0.0 vs -0.0
        return INFINITY;
    }
    
    // Calculate ULP distance
    int32_t gpu_signed, cpu_signed;
    std::memcpy(&gpu_signed, &gpu_bits, sizeof(int32_t));
    std::memcpy(&cpu_signed, &cpu_bits, sizeof(int32_t));
    
    // If signs are negative, flip all bits to get proper ordering
    if (gpu_signed < 0) gpu_signed = 0x80000000 - gpu_signed;
    if (cpu_signed < 0) cpu_signed = 0x80000000 - cpu_signed;
    
    return static_cast<float>(std::abs(gpu_signed - cpu_signed));
}





int main() {
    // Test values including the problematic one from softmax
    float test_values[] = {
        1.1422761679f,  // The problematic value from forward pass
        1.0f,           // log(1) = 0
        10.0f,          // log(10) = 2.302585...
        0.5f,           // log(0.5) = -0.693147...
        100.0f,
        1.5f
    };
    int count = sizeof(test_values) / sizeof(float);
    
    // Allocate device memory
    float *d_results;
    float *results = (float*)malloc(count * sizeof(float));
    HIP_CHECK(hipMalloc(&d_results, sizeof(float)));
    
    // Test log()
    printf("Testing log() function:\n");
    printf("%-20s %-25s %-25s %-15s %-10s\n", "Input", "GPU Result", "CPU Reference", "ULP Error", "Status");
    printf("===================================================================================\n");
    
    int log_pass = 1;
    for (int i = 0; i < count; ++i) {
        hipLaunchKernelGGL(test_log, dim3(1), dim3(1), 0, 0, d_results, test_values[i]);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(&results[i], d_results, sizeof(float), hipMemcpyDeviceToHost));
        
        float cpu_ref = std::log(test_values[i]);
        float ulp = ulp_error(results[i], cpu_ref);
        int pass = (ulp < 1.0f) ? 1 : 0;
        log_pass = log_pass && pass;
        
        printf("%-20.10f %-25.16f %-25.16f %-15.1f %s\n", 
               test_values[i], results[i], cpu_ref, ulp, 
               pass ? "PASS" : "FAIL");
    }
    
    // Test log10()
    printf("\n\nTesting log10() function:\n");
    printf("%-20s %-25s %-25s %-15s %-10s\n", "Input", "GPU Result", "CPU Reference", "ULP Error", "Status");
    printf("===================================================================================\n");
    
    int log10_pass = 1;
    for (int i = 0; i < count; ++i) {
        hipLaunchKernelGGL(test_log10, dim3(1), dim3(1), 0, 0, d_results, test_values[i]);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(&results[i], d_results, sizeof(float), hipMemcpyDeviceToHost));
        
        float cpu_ref = std::log10(test_values[i]);
        float ulp = ulp_error(results[i], cpu_ref);
        int pass = (ulp < 1.0f) ? 1 : 0;
        log10_pass = log10_pass && pass;
        
        printf("%-20.10f %-25.16f %-25.16f %-15.1f %s\n", 
               test_values[i], results[i], cpu_ref, ulp, 
               pass ? "PASS" : "FAIL");
    }
    
    HIP_CHECK(hipFree(d_results));
    free(results);
    
    printf("\n\n========== TEST SUMMARY ==========\n");
    printf("log() test:   %s\n", log_pass ? "PASS" : "FAIL");
    printf("log10() test: %s\n", log10_pass ? "PASS" : "FAIL");
    printf("Overall:      %s\n", (log_pass && log10_pass) ? "PASS" : "FAIL");
    
    return (log_pass && log10_pass) ? 0 : 1;
}
