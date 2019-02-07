
#include <stdio.h>
#include <omp.h>

#include "../utilities/check.h"
#include "../utilities/utilities.h"

#define TRIALS (1)

#define N (992)

#define INIT() INIT_LOOP(N, {C[i] = 1; D[i] = i; E[i] = -i;})

#define ZERO(X) ZERO_ARRAY(N, X) 

int main(void) {
  check_offloading();

  double A[N], B[N], C[N], D[N], E[N];

  INIT();

  //
  // Test: omp_get_thread_num()
  //
  ZERO(A);
  TEST({
    // Master in the serial section has thread id 0.
    int tid = omp_get_thread_num();
    A[tid] += tid;
    // Expecting to start 128 parallel threads.
  _Pragma("omp parallel num_threads(128)")
    {
      // Workers in parallel section have thread ids 0 ... 223
      int tid = omp_get_thread_num();
      A[tid] += tid;
    }
  }, VERIFY(0, 128, A[i], i*(trial+1)));

  //
  // Test: Execute parallel on device
  //
  TEST({
  _Pragma("omp parallel num_threads(128)")
    {
      int i = omp_get_thread_num()*4;
      for (int j = i; j < i + 4; j++) {
        B[j] = D[j] + E[j];
      }
    }
    }, VERIFY(0, 512, B[i], (double)0));

  //
  // Test: if clause serial execution of parallel region on target
  //
  ZERO(A);
  TEST({
  _Pragma("omp parallel num_threads(128) if(0)")
    {
      int tid = omp_get_thread_num();
      A[tid] = tid;
    }
  }, VERIFY(0, 128, A[i], 0));

  //
  // Test: if clause serial execution of parallel region on target
  //
  ZERO(A);
  TEST({
  _Pragma("omp parallel num_threads(128) if(A[1] == 1)")
    {
      int tid = omp_get_thread_num();
      A[tid] = tid;
    }
  }, VERIFY(0, 128, A[i], 0));

  //
  // Test: if clause parallel execution of parallel region on target
  //
  ZERO(A);
  TEST({
  _Pragma("omp parallel num_threads(128) if(A[1] == 0)")
    {
      int tid = omp_get_thread_num();
      A[tid] = tid;
    }
  }, VERIFY(0, 128, A[i], i));

  //
  // Test: proc_bind clause
  //
  TEST({
  _Pragma("omp parallel num_threads(128) proc_bind(master)")
    {
      int i = omp_get_thread_num()*4;
      for (int j = i; j < i + 4; j++) {
        B[j] = 1 + D[j] + E[j];
      }
    }
  _Pragma("omp parallel num_threads(128) proc_bind(close)")
    {
      int i = omp_get_thread_num()*4;
      for (int j = i; j < i + 4; j++) {
        B[j] += 1 + D[j] + E[j];
      }
    }
  _Pragma("omp parallel num_threads(128) proc_bind(spread)")
    {
      int i = omp_get_thread_num()*4;
      for (int j = i; j < i + 4; j++) {
        B[j] += 1 + D[j] + E[j];
      }
    }
  }, VERIFY(0, 512, B[i], 3));

  //
  // Test: num_threads on parallel.
  // We assume a maximum of 128 threads in the parallel region (32 are
  // reserved for the master warp).
  //
  // This test fails on Volta because a parallel region can only contain
  // <=32 or a multiple of 32 workers.
  for (int t = 1; t <= 128; t += t < 32 ? 1 : 32) {
    ZERO(A);
    int threads[1]; threads[0] = t;
    TEST({
    _Pragma("omp parallel num_threads(threads[0]) if(1)")
      {
        int tid = omp_get_thread_num();
        A[tid] = 99;
      }
    }, VERIFY(0, 128, A[i], 99*(i < t)));
  }

  //
  // Test: sharing of variables from master to parallel region.
  // FIXME: Currently we don't have support to share variables from
  // master to workers, so we're doing "serialized" parallel execution.
  //
  ZERO(A);
  TEST({
  double tmp = 1;
  A[0] = tmp;
  _Pragma("omp parallel if(0)")
    {
      tmp = 2;
      A[0] += tmp;
    }
  A[0] += tmp;
  }, VERIFY(0, 1, A[i], 5));

  //
  // Test: private clause on parallel region.
  // FIXME: Currently we don't have support to share variables from
  // master to workers, so we're doing "serialized" parallel execution.
  //
  ZERO(A);
  TEST({
  double tmp = 1;
  A[0] = tmp;
  _Pragma("omp parallel private(tmp) if(0)")
    {
      tmp = 2;
      A[0] += tmp;
    }
  A[0] += tmp;
  }, VERIFY(0, 1, A[i], 4));

  //
  // Test: firstprivate clause on parallel region.
  // FIXME: Currently we don't have support to share variables from
  // master to workers, so we're doing "serialized" parallel execution.
  //
  ZERO(A);
  TEST({
  double tmp = 1;
  A[0] = tmp;
  _Pragma("omp parallel firstprivate(tmp) if(0)")
    {
      tmp += 2;
      A[0] += tmp;
    }
  A[0] += tmp;
  }, VERIFY(0, 1, A[i], 5));

  //
  // Test: shared clause on parallel region.
  // FIXME: Currently we don't have support to share variables from
  // master to workers, so we're doing "serialized" parallel execution.
  //
  ZERO(A);
  TEST({
  double tmp = 1;
  A[0] = tmp;
  double distance = 21;
  _Pragma("omp parallel firstprivate(tmp) shared(distance) if(0)")
    {
      distance += 9;
      tmp += 2 + distance;
      A[0] += tmp;
    }
  A[0] += tmp + distance;
  }, VERIFY(0, 1, A[i], 65));

  //
  // Test: sharing of array from master to parallel region.
  //
  ZERO(A);
  ZERO(B);
  TEST({
  for (int i = 0; i < 128; i++) {
    B[i] = 0;
    A[i] = 99 + B[i];
    B[i] = 1;
  }
  _Pragma("omp parallel num_threads(128)")
    {
      int tid = omp_get_thread_num();
      A[tid] += 1;
      B[tid] += 2;
    }
  for (int i = 0; i < 128; i++) {
    A[i] += B[i];
  }
  }, VERIFY(0, 128, A[i], 103));

  //
  // Test: array private clause on parallel region.
  //
  ZERO(A);
  ZERO(B);
  TEST({
  for (int i = 0; i < 128; i++) {
    B[i] = 0;
    A[i] = 99 + B[i];
    B[i] = 1;
  }
  _Pragma("omp parallel num_threads(128) private(B) if(1)")
    {
      int tid = omp_get_thread_num();
      A[tid] += 1;
      B[tid] = 2;
    }
  for (int i = 0; i < 128; i++) {
    A[i] += B[i];
  }
  }, VERIFY(0, 128, A[i], 101));

  //
  // Test: array firstprivate clause on parallel region.
  //
  ZERO(A);
  ZERO(B);
  TEST({
  for (int i = 0; i < 128; i++) {
    B[i] = 0;
    A[i] = 99 + B[i];
    B[i] = 2;
  }
  _Pragma("omp parallel num_threads(128) firstprivate(B)")
    {
      int tid = omp_get_thread_num();
      B[tid] += 8;
      A[tid] += B[tid];
    }
  for (int i = 0; i < 128; i++) {
    A[i] += B[i];
  }
  }, VERIFY(0, 128, A[i], 111));

  //
  // Test: array shared clause on parallel region.
  // FIXME: Currently we don't have support to share variables from
  // master to workers, so we're doing "serialized" parallel execution.
  //
  ZERO(A);
  ZERO(B);
  TEST({
  B[0] = 0;
  A[0] = 99 + B[0];
  B[0] = 2;
  double distance[32];
  distance[30] = 21;
  _Pragma("omp parallel firstprivate(B) shared(distance, A) if(0)")
    {
      distance[30] += 9;
      B[0] += 8;
      A[0] += B[0] + distance[30];
    }
  A[0] += B[0] + distance[30];
  }, VERIFY(0, 1, A[i], 171));

  struct CITY {
    char name[128];
    int distance_to_nyc;
  };

  struct CONTEXT {
    struct CITY city;
    double A[N];
    double B[N];
  };

  struct CONTEXT data;

  //
  // Test: omp_get_thread_num()
  //
  strcpy(data.city.name, "dobbs ferry");
  data.city.distance_to_nyc = 21;
  ZERO(data.A);
  TEST({
    // Master in the serial section has thread id 0.
    int tid = omp_get_thread_num();
    data.A[tid] += tid + (int) data.city.name[1] + data.city.distance_to_nyc;
    // Expecting to start 128 parallel threads.
  _Pragma("omp parallel num_threads(128)")
    {
      // Workers in parallel section have thread ids 0 ... 127
      int tid = omp_get_thread_num();
      data.A[1+tid] += 1+tid + (int) data.city.name[1] + data.city.distance_to_nyc;
    }
  }, VERIFY(0, 128, data.A[i], (132 + i)*(trial+1)));

  //
  // Test: sharing of struct from master to parallel region.
  //
  ZERO(data.A);
  ZERO(data.B);
  TEST({
  for (int i = 0; i < 128; i++) {
    data.B[i] = 0;
    data.A[i] = 99 + data.B[i];
    data.B[i] = 1;
  }
  _Pragma("omp parallel num_threads(128)")
    {
      int tid = omp_get_thread_num();
      data.A[tid] += 1;
      data.B[tid] += 2;
    }
  for (int i = 0; i < 128; i++) {
    data.A[i] += data.B[i];
  }
  }, VERIFY(0, 128, data.A[i], 103));

  //
  // Test: struct private clause on parallel region.
  //
  ZERO(data.A);
  ZERO(data.B);
  TEST({
  for (int i = 0; i < 128; i++) {
    data.B[i] = 0;
    data.A[i] = 99 + data.B[i];
    data.B[i] = 1;
  }
  _Pragma("omp parallel num_threads(128) private(data)")
    {
      int tid = omp_get_thread_num();
      data.A[tid] += 1;
      data.B[tid] = 2;
    }
  for (int i = 0; i < 128; i++) {
    data.A[i] += data.B[i];
  }
  }, VERIFY(0, 1, data.A[i], 100));

  //
  // Test: struct firstprivate clause on parallel region.
  // FIXME: Currently we don't have support to share variables from
  // master to workers, so we're doing "serialized" parallel execution.
  //
  ZERO(data.A);
  ZERO(data.B);
  TEST({
  data.B[0] = 0;
  data.A[0] = 99 + data.B[0];
  data.B[0] = 2;
  double tmp;
  _Pragma("omp parallel firstprivate(data) if(0)")
    {
      data.B[0] += 8;
      data.A[0] += data.B[0];
      tmp = data.A[0];
    }
  data.A[0] += data.B[0] + tmp;
  }, VERIFY(0, 1, data.A[i], 210));

  //
  // Test: struct shared clause on parallel region.
  // FIXME: Currently we don't have support to share variables from
  // master to workers, so we're doing "serialized" parallel execution.
  //
  ZERO(A);
  ZERO(B);
  TEST({
  B[0] = 0;
  A[0] = 99 + B[0];
  B[0] = 2;
  struct CITY city;
  city.distance_to_nyc = 21;
  _Pragma("omp parallel firstprivate(B) shared(city, A) if(0)")
    {
      city.distance_to_nyc += 9;
      B[0] += 8;
      A[0] += B[0] + city.distance_to_nyc;
    }
  A[0] += B[0] + city.distance_to_nyc;
  }, VERIFY(0, 1, A[i], 171));

  return 0;
}
