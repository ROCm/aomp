#include <stdio.h>
#include <omp.h>
#include "../utilities/check.h"

// Struct to mimic the existence of a device if offloading is disabled.
struct NoOffloadCtorDtor {
  bool RunDtor;

  NoOffloadCtorDtor(bool V) : RunDtor(V) {
  
    if (!offloading_disabled() || RunDtor)
      return;
   
    printf("CtorA: 1728 Device\n");
    printf("CtorB: 39304 Device\n");
    printf("CtorD: 123 Device\n");
    printf("CtorD: 123 Device\n");
    printf("CtorD: 123 Device\n");
  }
  
  ~NoOffloadCtorDtor() {
  
    if (!offloading_disabled() || !RunDtor)
      return;
    printf("DtorA: 5159780352 Device\n");
    printf("DtorC: 123 Device\n");
    printf("DtorD: 1860867 Device\n");
    printf("DtorD: 1860867 Device\n");
    printf("DtorD: 1860867 Device\n");
    printf("DtorE: 1860867 Device\n");
  }
};

NoOffloadCtorDtor NC0(true);

#pragma omp declare target

struct SSW {
  int A = 123;

  SSW(int B) {
    A = B*B/B;
    printf("CtorW: %d %s\n", A, omp_is_initial_device() ? "Host" : "Device");
  }

  ~SSW() {
    A *= A/A;
    printf("DtorW: %d %s\n",A, omp_is_initial_device() ? "Host" : "Device");
  }
};

struct SSA {
  long int A;

  SSA(int B) {
    A = B*B*B;
    printf("CtorA: %ld %s\n",A, omp_is_initial_device() ? "Host" : "Device");
  }
  ~SSA() {
    A *= A*A;
    printf("DtorA: %ld %s\n",A, omp_is_initial_device() ? "Host" : "Device");
  }
};

struct SSB {
  int A;

  SSB(int B) {
    A = B*B*B;
    printf("CtorB: %d %s\n",A, omp_is_initial_device() ? "Host" : "Device");
  }
};

struct SSC {
  ~SSC() {
    printf("DtorC: %d %s\n",123, omp_is_initial_device() ? "Host" : "Device");
  }
};

struct SSD {
  int A;

  SSD() {
    A = 123;
    printf("CtorD: %d %s\n",A, omp_is_initial_device() ? "Host" : "Device");
  }
  ~SSD() {
    A *= A*A;
    printf("DtorD: %d %s\n",A, omp_is_initial_device() ? "Host" : "Device");
  }
};
struct SSE {
  int A = 123;
  ~SSE() {
    A *= A*A;
    printf("DtorE: %d %s\n",A, omp_is_initial_device() ? "Host" : "Device");
  }
};

SSA sa(12);
SSB sb(34);
SSC sc;
SSD sd[3];
SSE se;

#pragma omp end declare target

SSW sw(56);

NoOffloadCtorDtor NC1(false);

int main(void) {

  bool OffloadDisabled = offloading_disabled();

  #pragma omp target device(0)
  #pragma omp teams num_teams(1) thread_limit(1)
  {
   printf("Main: %d %s\n",123, (omp_is_initial_device() && !OffloadDisabled) ? "Host" : "Device");
  }
  return 0;
}
