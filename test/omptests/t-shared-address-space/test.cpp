#if 0
#include <stdio.h>
#include <omp.h>

int main(int argc, char *argv[]) {
  int n=2500;
  
  int ch = 6;
  #pragma omp target
  #pragma omp teams
  #pragma omp distribute parallel for schedule(static,ch)
  for (int i = 0; i < 1024; i++) {
  }      
  return 0;
}
#endif

#if 1
#include <stdio.h>
#include <omp.h>

int main(int argc, char *argv[]) {
//   int n=2500;
//   double a[n];
// 
//   #pragma omp target device(1)                                                                                     
//   //#pragma omp teams num_teams(1) thread_limit(1)
//   //#pragma omp distribute 
//   #pragma omp parallel for
//   for (int i = 0; i < 32; ++i) {
//     a[i] += 1;
//   }


  #pragma omp target device(1)
  {
    static __attribute__((address_space(3))) int A;
    A = 123;
    
    #pragma omp parallel num_threads(2)
    {
      printf("%d\n",A);
    }
  }

  return 0;
}
#endif

#if 0
#include <stdio.h>
#include <omp.h>

#pragma omp declare target

// struct SSW {
//   int A = 123;
// 
//   SSW(int B) {
//     A = B*B/B;
//     printf("CtorW: %d %s\n", A, omp_is_initial_device() ? "Host" : "Device");
//   }
// 
//   ~SSW() {
//     A *= A/A;
//     printf("DtorW: %d %s\n",A, omp_is_initial_device() ? "Host" : "Device");
//   }
// };

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

// struct SSB {
//   int A;
// 
//   SSB(int B) {
//     A = B*B*B;
//     printf("CtorB: %d %s\n",A, omp_is_initial_device() ? "Host" : "Device");
//   }
// };
// 
// struct SSC {
//   ~SSC() {
//     printf("DtorC: %d %s\n",123, omp_is_initial_device() ? "Host" : "Device");
//   }
// };
// 
// struct SSD {
//   int A;
// 
//   SSD() {
//     A = 123;
//     printf("CtorD: %d %s\n",A, omp_is_initial_device() ? "Host" : "Device");
//   }
//   ~SSD() {
//     A *= A*A;
//     printf("DtorD: %d %s\n",A, omp_is_initial_device() ? "Host" : "Device");
//   }
// };
// struct SSE {
//   int A = 123;
//   ~SSE() {
//     A *= A*A;
//     printf("DtorE: %d %s\n",A, omp_is_initial_device() ? "Host" : "Device");
//   }
// };

SSA sa(12);
// SSB sb(34);
// SSC sc;
// SSD sd[3];
// SSE se;
#pragma omp end declare target

int A = 123;

// SSW sw(56);

int main(void) {
  #pragma omp target device(1)
  #pragma omp teams num_teams(1) thread_limit(1)
  {
   printf("Main: %d %s %d\n",123, omp_is_initial_device() ? "Host" : "Device", A);
  }
  return 0;
}

#pragma omp declare target link(A)
#endif 
