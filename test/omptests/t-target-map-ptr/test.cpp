#include <stdio.h>
#include <omp.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include "../utilities/check.h"
#include "../utilities/utilities.h"

// enable tests
#define CHECK              1  /* 1: */
#define FULL               1  /* 1: */
#define FULL_ZERO          1  /* 1: use zero ptrs */
#define FULL_ZERO_NULL     1  /* 0: use zero ptrs */
#define FULL_ZERO_IMPLICIT 1  /* 1: use zero ptrs, with some that should be null on device */
#define FULL_S             0  /* 0: need struct support */
#define OFFSET             1  /* 1: */
#define OFFSET_S           0  /* 0: need struct support */
#define FP                 1  /* 1: first-private arrays */
#define FP_RANGES          0  /* 0: first-private arrays with ranges, not supported yet */
#define FPPTR              0  /* 0: first-private arrays referenced via pointers, not supported yet */
#define DEVICEPTR          1  /* 1: is_device_ptr/use_device_ptr */
#define PTR_REF            0  /* 0: maps with references to pointers */

#define N (992)

#define INIT() INIT_LOOP(N, {A[i] = 0; C[i] = 1; D[i] = i; E[i] = -i;\
  s1.A[i] = 0; s1.C[i] = 1; s1.D[i] = i; s1.E[i] = -i; })

typedef struct S {
  double A[N], B[N], C[N], D[N], E[N];
  double *pA, *pB, *pC, *pD, *pE;
} S;

int main(void){
  #if CHECK
    check_offloading();
  #endif

  int fail;
  double A[N], B[N], C[N], D[N], E[N];
  double *pA, *pB, *pC, *pD, *pE;
  S s1;

  // map ptrs
  pA = &A[0];
  pB = &B[0];
  pC = &C[0];
  pD = &D[0];
  pE = &E[0];
  s1.pA = &s1.A[0];
  s1.pB = &s1.B[0];
  s1.pC = &s1.C[0];
  s1.pD = &s1.D[0];
  s1.pE = &s1.E[0];

  
#if FULL
  //
  // Test: Execute on device (full extent)
  //
  INIT();
  #pragma omp target device(1) map(from: pA[0:N]) map(to: pC[0:N]) map(pD[0:N])
  {
    #pragma omp parallel for schedule(static,1)
    for (int i = 0; i < 992; i++)
      pA[i] = pC[i] + pD[i] + 1 /*omp_is_initial_device()*/;
  }
  // CHECK: Succeeded
  fail = 0;
  VERIFY(0, N, A[i], (double)(i+2));
  if (fail) {
    printf ("Test full extent: Failed\n");
  } else {
    printf ("Test full extent: Succeeded\n");
  }
#endif

#if FULL_ZERO
  //
  // Test: Execute on device (full extent) with zero-length pointers
  //
  INIT();
  #pragma omp target data map(from: pA[0:N]) map(to: pC[0:N]) map(pD[0:N]) \
                          device(1)
  {
    // explicit implicit zero ptr
    #pragma omp target device(1) map(from: pA[0:0]) map(to: pC[0:0]) map(pD[:0])
    {
      #pragma omp parallel for schedule(static,1) 
      for (int i = 0; i < 992; i++)
        pA[i] = pC[i] + pD[i] + 1 /*omp_is_initial_device()*/;
    }
  }
  // CHECK: Succeeded
  fail = 0;
  VERIFY(0, N, A[i], (double)(i+2));
  if (fail) {
    printf ("Test full extent with zero length ptrs: Failed\n");
  } else {
    printf ("Test full extent with zero length ptrs: Succeeded\n");
  }
#endif

  #if FULL_ZERO_NULL
  //
  // Test: Execute on device (full extent) with zero-length pointers, with some that are
  // expected to be null
  //
  INIT();
  int isNull_b = 2; // b is not mapped at all; 
  int isNull_e = 2; // e is partially mapped, but not the part starting at zero offset
  #pragma omp target data map(from: pA[0:N]) map(to: pC[0:N]) map(pD[0:N], pE[10:10]) \
                          device(0)
  {
    // explicit implicit zero ptr
    #pragma omp target device(0) map(from: pA[0:0]) map(to: pC[0:0]) \
      map(pD[:0], pB[0:0], pE[0:0], isNull_b, isNull_e)
    {
      isNull_b = (pB == NULL) ? 1 : 0;
      isNull_e = (pE == NULL) ? 1 : 0;
      #pragma omp parallel for schedule(static,1) 
      for (int i = 0; i < 992; i++)
        pA[i] = pC[i] + pD[i] + 1 /*omp_is_initial_device()*/;
    }
  }
  // CHECK: Succeeded
  fail = 0;
  VERIFY(0, N, A[i], (double)(i+2));
  if (!offloading_disabled()) {
    if (isNull_b != 1) {
      printf("failed the null test for var B, got %d\n", isNull_b);
      fail++;
    }
    if (isNull_e != 1) {
      printf("failed the null test for var E, got %d\n", isNull_e);
      fail++;
    }
  }
  if (fail) {
    printf ("Test full extent with zero length ptrs and NULL ptr: Failed\n");
  } else {
    printf ("Test full extent with zero length ptrs and NULL ptr: Succeeded\n");
  }
#endif

#if FULL_ZERO_IMPLICIT
  //
  // Test: Execute on device (full extent) with implicit zero-length pointers
  //
  INIT();
  #pragma omp target data map(from: pA[0:N]) map(to: pC[0:N]) map(pD[0:N]) \
                          device(1)
  {
    #pragma omp target device(1) // implicit zero ptr
    {
      #pragma omp parallel for schedule(static,1) 
      for (int i = 0; i < 992; i++)
        pA[i] = pC[i] + pD[i] + 1 /*omp_is_initial_device()*/;
    }
  }
  // CHECK: Succeeded
  fail = 0;
  VERIFY(0, N, A[i], (double)(i+2));
  if (fail) {
    printf ("Test full extent with implicit zero length ptrs: Failed\n");
  } else {
    printf ("Test full extent with implicit zero length ptrs: Succeeded\n");
  }
#endif

#if FULL_S
  //
  // Test: Execute on device (full extent)
  //
  INIT();
  #pragma omp target data device(1) map(s1)
  #pragma omp target device(1) map(from: s1.pA[0:N]) map(to: s1.pC[0:N]) \
                               map(s1.pD[0:N])
  {
    #pragma omp parallel for schedule(static,1)
    for (int i = 0; i < 992; i++)
      s1.pA[i] = s1.pC[i] + s1.pD[i] + 1 /*omp_is_initial_device()*/;
  }
  // CHECK: Succeeded
  fail = 0;
  VERIFY(0, N, s1.A[i], (double)(i+2));
  if (fail) {
    printf ("Test full extent struct: Failed\n");
  } else {
    printf ("Test full extent struct: Succeeded\n");
  }
#endif

#if OFFSET
  //
  // Test: Execute on device (with offsets)
  //
  pA = pA - 100;
  pC = pC - 200;
  pD = pD - 300;
  INIT();
  #pragma omp target device(1) map(from: pA[100:N]) map(to: pC[200:N]) \
                               map(pD[300:N])
  {
    #pragma omp parallel for schedule(static,1)
    for (int i = 0; i < 992; i++)
      pA[i+100] = pC[i+200] + pD[i+300] + 1 /*omp_is_initial_device()*/;
  }
  // CHECK: Succeeded
  fail = 0;
  VERIFY(0, N, A[i], (double)(i+2));
  if (fail) {
    printf ("Test Offset: Failed\n");
  } else {
    printf ("Test Offset: Succeeded\n");
  }

  // Restore original pointers
  pA = &A[0];
  pC = &C[0];
  pD = &D[0];
#endif

#if OFFSET_S
  //
  // Test: Execute on device (full extent)
  //
  s1.pA = s1.pA - 100;
  s1.pC = s1.pC - 200;
  s1.pD = s1.pD - 300;
  INIT();
  #pragma omp target device(1) map(s1) map(from: s1.pA[100:N]) \
                               map(to: s1.pC[200:N]) map(s1.pD[300:N])
  {
    #pragma omp parallel for schedule(static,1)
    for (int i = 0; i < 992; i++)
      s1.pA[i] = s1.pC[i] + s1.pD[i] + 1 /*omp_is_initial_device()*/;
  }
  // CHECK: Succeeded
  fail = 0;
  VERIFY(0, N, s1.A[i], (double)(i+2));
  if (fail) {
    printf ("Test Offset struct: Failed\n");
  } else {
    printf ("Test Offset struct: Succeeded\n");
  }
#endif

#if FP
  //
  // Test: Execute on device with first-private arrays
  //
  INIT();
  #pragma omp target device(1) map(from: A[0:N]) firstprivate(C, D)
  {
    #pragma omp parallel for schedule(static,1)
    for (int i = 0; i < 992; i++)
      A[i] = C[i] + D[i] + 1 /*omp_is_initial_device()*/;
  }
  // CHECK: Succeeded
  fail = 0;
  VERIFY(0, N, A[i], (double)(i+2));
  if (fail) {
    printf ("Test first-private: Failed\n");
  } else {
    printf ("Test first-private: Succeeded\n");
  }
#endif

#if FP_RANGES
  //
  // Test: Execute on device with first-private arrays with ranges
  //
  INIT();
  #pragma omp target device(1) map(from: A[0:N]) firstprivate(C[0:N], D[0:N])
  {
    #pragma omp parallel for schedule(static,1)
    for (int i = 0; i < 992; i++)
      A[i] = C[i] + D[i] + 1 /*omp_is_initial_device()*/;
  }
  // CHECK: Succeeded
  fail = 0;
  VERIFY(0, N, A[i], (double)(i+2));
  if (fail) {
    printf ("Test first-private with ranges: Failed\n");
  } else {
    printf ("Test first-private with ranges: Succeeded\n");
  }
#endif

#if FPPTR
  //
  // Test: Execute on device with first-private arrays referenced via pointers
  //
  INIT();
  #pragma omp target device(1) map(from: pA[0:N]) firstprivate(pC[0:N], pD[0:N])
  {
    #pragma omp parallel for schedule(static,1)
    for (int i = 0; i < 992; i++)
      pA[i] = pC[i] + pD[i] + 1 /*omp_is_initial_device()*/;
  }
  // CHECK: Succeeded
  fail = 0;
  VERIFY(0, N, A[i], (double)(i+2));
  if (fail) {
    printf ("Test first-private with pointers: Failed\n");
  } else {
    printf ("Test first-private with pointers: Succeeded\n");
  }
#endif

#if DEVICEPTR
  //
  // Test: Execute on device with device pointers
  //
  INIT();

  #pragma omp target data device(1) map(from: pA[0:N]) map(to: pC[0:N]) \
                     use_device_ptr(pC)
  {
    // Populate device pC with the contents of host pD
    cudaMemcpy(pC, pD, N*sizeof(double), cudaMemcpyHostToDevice);

    #pragma omp target device(1) is_device_ptr(pC)
    {
      #pragma omp parallel for schedule(static,1)
      for (int i = 0; i < 992; i++)
        pA[i] = pC[i] + (i-1)*omp_is_initial_device() + 1;
      // pC on the device was populated with the contents of pD, so:
      // if offloading succeeds: pA[i] = i + (i-1)*0 + 1 = i+1
      // if offloading fails:    pA[i] = 1 + (i-1)*1 + 1 = i+1
    }
  }

  // CHECK: Succeeded
  fail = 0;
  VERIFY(0, N, A[i], (double)(i+1));
  if (fail) {
    printf ("Test device_ptr: Failed\n");
  } else {
    printf ("Test device_ptr: Succeeded\n");
  }
#endif

#if PTR_REF
  //
  // Test: Mappings with references to pointers
  //
  double*& rpA = pA;
  double*& rpB = pB;
  double*& rpC = pC;
  double*& rpD = pD;
  double*& rpE = pE;

  INIT();
  #pragma omp target map(from: rpA[0:N]) map(to: rpC[0:N]) map(rpD[0:N]) \
                     device(1)
  {
    #pragma omp parallel for schedule(static,1)
    for (int i = 0; i < 992; i++)
      rpA[i] = rpC[i] + rpD[i] + 1 /*omp_is_initial_device()*/;
  }
  // CHECK: Succeeded
  fail = 0;
  VERIFY(0, N, A[i], (double)(i+2));
  if (fail) {
    printf ("Test refs to ptrs: Failed\n");
  } else {
    printf ("Test refs to ptrs: Succeeded\n");
  }
#endif

  return 0;
}
