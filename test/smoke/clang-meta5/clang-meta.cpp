#include <omp.h>
#include <stdio.h>


#if defined(SPEC_OPENMP_INNER_SIMD)
#define OMP_MODEL_TARGET        0
#define OMP_MODEL_TARGET_SIMD   1
#define OMP_MODEL_LOOP          0
#elif defined(SPEC_OPENMP_TARGET)
#define OMP_MODEL_TARGET        1
#define OMP_MODEL_TARGET_SIMD   0
#define OMP_MODEL_LOOP          0
#else
// default is to use the target teams loop version
#define OMP_MODEL_TARGET        0
#define OMP_MODEL_TARGET_SIMD   0
#define OMP_MODEL_LOOP          1
#endif

const int nx=2; const int ny=2; const int nz=2;

int main() {
  int i, j, k;
#pragma omp metadirective when(user={condition(OMP_MODEL_TARGET)}: target teams distribute parallel for simd collapse(3)) \
                          when(user={condition(OMP_MODEL_TARGET_SIMD)}: target teams distribute parallel for private(i) collapse(2)) \
                          when(user={condition(OMP_MODEL_LOOP)}: target teams loop collapse(2)) 
  for(k=0;k<nz;k++) {
    for(j=0;j<ny;j++) {
#pragma omp metadirective when(user={condition(OMP_MODEL_TARGET_SIMD)}: simd) \
                          when(user={condition(OMP_MODEL_LOOP)}: loop ) \
                          when(user={condition(OMP_MODEL_TARGET)}: nothing )
      for(i=0;i<nx;i++)
        printf("hello\n");
    }
  }
  return 0;
}

