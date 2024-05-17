#include <cstdio>
#include <new>
#include "hip/hip_runtime.h"

struct b {
#if 0
  __host__
  __device__
#endif
  virtual void my_print() {
    printf("Hello\n");
  }
};

struct d : b {
#if 0
  __host__ 
  __device__
#endif
  virtual void my_print() override {
    printf("World\n");
  }
};



int main() {

    constexpr int num_objects = 2;
    void** buffers{nullptr};
    hipHostMalloc(&buffers, num_objects * sizeof(void*));
    for (auto i = 0; i < num_objects; ++i) {
      hipMalloc(buffers + i, sizeof(d));
    }
    
    #pragma omp target teams distribute parallel for is_device_ptr(buffers)
    for (int i = 0; i < num_objects; ++i) {
        if (i % 2 == 0) {
            new(*(buffers + i)) b();
        }
        else {
            new(*(buffers + i)) d();
        }
    }

    #pragma omp target teams distribute parallel for is_device_ptr(buffers)
    for (int i = 0; i < num_objects; ++i) {
        static_cast<b*>(*(buffers + i))->my_print();
    }
#pragma omp target 
    {
      printf("last region\n");
    }

    for (auto i = 0; i < num_objects; ++i) {
        hipFree(*(buffers + i));
    }
    hipHostFree(buffers);
    return 0;
}
