#include <chrono>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <time.h>
#include <vector>

using namespace std;
using namespace std::chrono;

template <typename T> void saxpy_serial(T *x, T *y, T a, size_t n) {
  for (size_t i = 0; i < n; ++i)
    x[i] = a * x[i] + y[i];
}

template <typename T> void saxpy_omp_cpu(T *x, T *y, T a, size_t n) {
#pragma omp parallel for
  for (size_t i = 0; i < n; ++i)
    x[i] = a * x[i] + y[i];
}

template <typename T> void saxpy_omp_gpu(T *x, T *y, T a, size_t n) {
#pragma omp target teams distribute parallel for map(tofrom                    \
                                                     : x [0:n]) map(to         \
                                                                    : y [0:n])
  for (size_t i = 0; i < n; ++i)
    x[i] = a * x[i] + y[i];
}

template <typename T> void saxpy_omp_gpu_loop(T *x, T *y, T a, size_t n) {
// would lie to use loop #pragma omp target parallel loop map(tofrom : x [0:n]) map(to : y [0:n])
#pragma omp target parallel for map(tofrom : x [0:n]) map(to : y [0:n])
  for (size_t i = 0; i < n; ++i)
    x[i] = a * x[i] + y[i];
}

template <typename T> bool equalQ(T *x, T *y, size_t n) {
  for (size_t i = 0; i < n; ++i)
    if (x[i] != y[i])
      return false;
  return true;
}

inline nanoseconds now() {
  timespec tm;
  clock_gettime(CLOCK_MONOTONIC, &tm);
  return seconds(tm.tv_sec) + nanoseconds(tm.tv_nsec);
}

string to_string(nanoseconds ns) {
  ostringstream os;
  os.fill('0');
  auto m = duration_cast<minutes>(ns);
  ns -= m;
  auto s = duration_cast<seconds>(ns);
  ns -= s;
  os << setw(2) << m.count() << ":" << setw(2) << s.count() << "." << setw(9)
     << ns.count();
  return os.str();
}

#define TIME(str, x)                                                           \
  {                                                                            \
    auto t1 = now();                                                           \
    x;                                                                         \
    auto t2 = now();                                                           \
    cout << str << to_string(t2 - t1) << endl;                                 \
  }

int main(int argc, char **argv) {
  int n = 500000000;
  double a = 3.;
  vector<double> x(n, 1.), y(n, 2.);
  vector<double> xcpu(x), xgpu(x), xloop(x);
  TIME("Serial time: ", saxpy_serial(x.data(), y.data(), a, n));
  TIME("OMP CPU time: ", saxpy_omp_cpu(xcpu.data(), y.data(), a, n));
  TIME("OMP GPU time: ", saxpy_omp_gpu(xgpu.data(), y.data(), a, n));
  TIME("OMP GPU loop time: ", saxpy_omp_gpu_loop(xloop.data(), y.data(), a, n));
  cout << "Serial == OMP CPU: " << equalQ(x.data(), xcpu.data(), n) << endl;
  cout << "Serial == OMP GPU: " << equalQ(x.data(), xgpu.data(), n) << endl;
  cout << "Serial == OMP GPU loop: " << equalQ(x.data(), xloop.data(), n) << endl;
  if (equalQ(x.data(), xloop.data(), n) != 1) return 1;
   
}
