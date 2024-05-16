 #include <chrono>
 #include <iomanip>
 #include <iostream>
 #include "omp.h"

/**
 * This may be a buggy program.
 * mapping the const global variables as tofrom is, strictly speaking wrong.
 * At least imho
 */
const int N = 2048;
const int K = 2048;
const int L = 2048;
const int M = 2048;
const int O = 2048;
const int P = 2048;
const int Q = 2048;

struct MiniRAIITimer {
  explicit MiniRAIITimer(const std::string regName)
      : regionName(regName), start(omp_get_wtime()) {}
  ~MiniRAIITimer() {
    double end = omp_get_wtime();
    double diff = end - start;
    std::cout << std::setw(9) << "The OpenMP region that maps " << regionName
              << " took: " << diff << " seconds" << std::endl;
  }

  std::string regionName;
  double start;
};

int main(int argc, char **argv) {

  std::cout << "OMP target map benchmark" << std::endl;
  {
    int tmp = 0;
    #pragma omp target teams distribute firstprivate(tmp) map(to:N)
    for (int i = 0; i < N; ++i) {
      tmp += i;
    }
  }

  {
    MiniRAIITimer timer("1 global");

    int tmp = 0;
    for (int I = 0; I < 10000; ++I) {
      /** Benchmark Region Start */
      #pragma omp target teams distribute firstprivate(tmp) map(to:N)
      for (int i = 0; i < N; ++i) {
        tmp += i;
      }
      /** Benchmark Region End */
    }
  }
  {
    MiniRAIITimer timer("2 globals");

    int tmp = 0;
    for (int I = 0; I < 10000; ++I) {
      /** Benchmark Region Start */
      #pragma omp target teams distribute firstprivate(tmp) map(to:N, M)
      for (int i = 0; i < N; ++i) {
        tmp += i;
      }
      /** Benchmark Region End */
    }
  }
  {
    MiniRAIITimer timer("3 globals");

    int tmp = 0;
    for (int I = 0; I < 10000; ++I) {
      /** Benchmark Region Start */
      #pragma omp target teams distribute firstprivate(tmp) map(to:N, M, K)
      for (int i = 0; i < N; ++i) {
        tmp += i;
      }
      /** Benchmark Region End */
    }
  }
  {
    MiniRAIITimer timer("4 globals");

    int tmp = 0;
    for (int I = 0; I < 10000; ++I) {
      /** Benchmark Region Start */
      #pragma omp target teams distribute firstprivate(tmp) map(to:N, M, K, L)
      for (int i = 0; i < N; ++i) {
        tmp += i;
      }
      /** Benchmark Region End */
    }
  }
  {
    MiniRAIITimer timer("5 globals");

    int tmp = 0;
    for (int I = 0; I < 10000; ++I) {
      /** Benchmark Region Start */
      #pragma omp target teams distribute firstprivate(tmp) map(to:N, M, K, L, O)
      for (int i = 0; i < N; ++i) {
        tmp += i;
      }
      /** Benchmark Region End */
    }
  }
  {
    MiniRAIITimer timer("6 globals");

    int tmp = 0;
    for (int I = 0; I < 10000; ++I) {
      /** Benchmark Region Start */
      #pragma omp target teams distribute firstprivate(tmp) map(to:N, M, K, L, O, P)
      for (int i = 0; i < N; ++i) {
        tmp += i;
      }
      /** Benchmark Region End */
    }
  }

  {
    MiniRAIITimer timer("7 globals");
    int tmp = 0;
    for (int I = 0; I < 10000; ++I) {
      /** Benchmark Region Start */
      #pragma omp target teams distribute firstprivate(tmp) map(to:N, M, K, L, O, P, Q)
      for (int i = 0; i < N; ++i) {
        tmp += i;
      }
      /** Benchmark Region End */
    }
  }

  std::cerr << "Passed" << std::endl;

  return 0;
}
