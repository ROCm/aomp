#include <stdio.h>
#include <omp.h>
#define N 10
#define GPU_THREAD_COUNT 256

int check_device_kind_gpu_selector() {
  int threadCount = 0;

  #pragma omp target map(tofrom: threadCount)
  {
    #pragma omp metadirective          \
      when(device = {kind(gpu)}: parallel) \
      default(single)

    threadCount = omp_get_num_threads();
  }

  if (threadCount != GPU_THREAD_COUNT) {
    printf("Failed metadirective: device_kind_gpu_selector\n");
    return 0;
  }
  return 1;
}

int check_device_kind_cpu_selector() {
  int threadCount = 0;

  #pragma omp target map(tofrom: threadCount)
  {
    #pragma omp metadirective          \
      when(device = {kind(cpu, host)}: parallel) \
      default(single)

    threadCount = omp_get_num_threads();
  }

  if (threadCount != 1) {
    printf("Failed metadirective: device_kind_cpu_selector\n");
    return 0;
  }
  return 1;
}

int check_device_arch_amdgcn_selector() {
  int threadCount = 0;

  #pragma omp target map(tofrom: threadCount)
  {
    #pragma omp metadirective          \
      when(device = {arch("amdgcn")}: parallel) \
      default(single)

    threadCount = omp_get_num_threads();
  }

  if (threadCount != GPU_THREAD_COUNT) {
    printf("Failed metadirective: device_arch_amdgcn_selector\n");
    return 0;
  }
  return 1;
}

int check_device_arch_x86_64_selector() {
  int threadCount = 0;

  #pragma omp target map(tofrom: threadCount)
  {
    #pragma omp metadirective          \
      when(device = {arch("x86_64")}: parallel) \
      default(single)

    threadCount = omp_get_num_threads();
  }

  if (threadCount != 1) {
    printf("Failed metadirective: device_arch_x86_64_selector\n");
    return 0;
  }
  return 1;
}

int check_device_isa_feature_selector() {
  int threadCount = 0;

  #pragma omp target map(tofrom: threadCount)
  {
    #pragma omp metadirective          \
      when(device = {isa("dl-insts")}: parallel) \
      default(single)

    threadCount = omp_get_num_threads();
  }

  if (threadCount != GPU_THREAD_COUNT) {
    printf("Failed metadirective: device_isa_feature_selector\n");
    return 0;
  }
  return 1;
}

int check_implementation_vendor_selector() {
  int threadCount = 0;

  #pragma omp target map(tofrom: threadCount)
  {
    #pragma omp metadirective          \
      when(implementation = {vendor(amd)}: parallel) \
      default(single)

    threadCount = omp_get_num_threads();
  }

  if (threadCount != GPU_THREAD_COUNT) {
    printf("Failed metadirective: implementation_vendor_selector\n");
    return 0;
  }
  return 1;
}

int check_scoring() {
  int threadCount = 0;

  #pragma omp target map(tofrom: threadCount)
  {
    #pragma omp metadirective          \
      when(implementation = {vendor(score(20): amd)}: parallel num_threads(4))\
      when(implementation = {vendor(score(100): amd)}: parallel num_threads(8))\
      default(single)

    threadCount = omp_get_num_threads();
  }

  if (threadCount > 8) {
    printf("Failed metadirective: scoring\n");
    return 0;
  }
  return 1;
}

int check_extension_match_any() {
  int threadCount = 0;

  #pragma omp target map(tofrom: threadCount)
  {
    #pragma omp metadirective          \
      when(device = {kind(cpu), arch("amdgcn")}, \
           implementation = {extension(match_any)} \
      : parallel)\
      default(single)
    threadCount = omp_get_num_threads();
  }

  if (threadCount != GPU_THREAD_COUNT) {
    printf("Failed metadirective: check_extension_match_any\n");
    return 0;
  }
  return 1;
}

int check_extension_match_all() {
  int threadCount = 0;

  #pragma omp target map(tofrom: threadCount)
  {
    #pragma omp metadirective          \
      when(device = {kind(cpu), arch("amdgcn")}, \
           implementation = {extension(match_all)} \
      : parallel)\
      default(single)
    threadCount = omp_get_num_threads();
  }

  if (threadCount != 1) {
    printf("Failed metadirective: check_extension_match_all\n");
    return 0;
  }
  return 1;
}

int check_static_condition_selector() {
  int threadCount = 0;

  #pragma omp target map(tofrom: threadCount)
  {
    #pragma omp metadirective          \
      when(user = {condition(N > 5)}: parallel num_threads(4)) \
      default(single)

    threadCount = omp_get_num_threads();
  }

  if (threadCount > 4) {
    printf("Failed metadirective: static_condition_selector\n");
    return 0;
  }
  return 1;
}

int main(void) {

  if (!check_device_kind_gpu_selector() ||
      !check_device_kind_cpu_selector() ||
      !check_device_arch_amdgcn_selector() ||
      !check_device_arch_x86_64_selector() ||
      !check_device_isa_feature_selector() ||
      !check_implementation_vendor_selector() ||
      !check_scoring() ||
      !check_extension_match_any() ||
      !check_extension_match_all() ||
      !check_static_condition_selector()) {
    return -1;
  }

  printf("Success\n");
  return 0;
}
