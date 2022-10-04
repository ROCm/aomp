#include <cstdlib> // For aligned_alloc
#include <omp.h>
#include <vector>

#ifndef ALIGNMENT
#define ALIGNMENT (2 * 1024 * 1024) // 2MB
#endif

#include <iostream>
#include <stdexcept>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <numeric>
#include <vector>

#include "DeviceDeclares.h"
#include "HostDefs.h"

#define _XTEAM_NUM_THREADS 1024

const int ARRAY_SIZE = 33554432;
// const int ARRAY_SIZE = 33;
unsigned int num_times = 12;
unsigned int ignore_num_times = 2;
unsigned int smoke_rc = 0;

template <typename T, bool> void run_kernels(const int ARRAY_SIZE);
template <typename TC, typename T> void run_kernels_complex(const int ARRAY_SIZE);

int main(int argc, char *argv[]) {
  std::cout << std::endl << "TEST DOUBLE" << std::endl;
  run_kernels<double, false>(ARRAY_SIZE);
  std::cout << std::endl << "TEST FLOAT" << std::endl;
  run_kernels<float, false>(ARRAY_SIZE);
  std::cout << std::endl << "TEST INT" << std::endl;
  run_kernels<int, true>(ARRAY_SIZE);
  std::cout << std::endl << "TEST UNSIGNED INT" << std::endl;
  run_kernels<unsigned, true>(ARRAY_SIZE);
  std::cout << std::endl << "TEST LONG" << std::endl;
  run_kernels<long, true>(ARRAY_SIZE);
  std::cout << std::endl << "TEST UNSIGNED LONG" << std::endl;
  run_kernels<unsigned long, true>(ARRAY_SIZE);
  std::cout << std::endl << "TEST FLOAT COMPLEX" << std::endl;
  run_kernels_complex<float _Complex, float>(ARRAY_SIZE);
  //std::cout << std::endl << "TEST DOUBLE COMPLEX" << std::endl;
  //run_kernels_complex<double _Complex, double>(ARRAY_SIZE);
  return smoke_rc;
}

// ---- Local overloads for testing sum
void __attribute__((flatten, always_inline))
sum_local_overload(double val, double *rval, double *xteam_mem,
                   unsigned int *td_ptr, const double initval) {
  __kmpc_xteamr_d_16x64(val, rval, xteam_mem, td_ptr, __kmpc_rfun_sum_d,
                        __kmpc_rfun_sum_lds_d, initval);
}
void __attribute__((flatten, always_inline))
sum_local_overload(float val, float *rval, float *xteam_mem,
                   unsigned int *td_ptr, const float initval) {
  __kmpc_xteamr_f_16x64(val, rval, xteam_mem, td_ptr, __kmpc_rfun_sum_f,
                        __kmpc_rfun_sum_lds_f, initval);
}
void __attribute__((flatten, always_inline))
sum_local_overload(double _Complex val, double _Complex *rval, double _Complex *xteam_mem,
                   unsigned int *td_ptr, const double _Complex initval) {
  __kmpc_xteamr_cd_16x64(val, rval, xteam_mem, td_ptr, __kmpc_rfun_sum_cd,
                        __kmpc_rfun_sum_lds_cd, initval);
}
void __attribute__((flatten, always_inline))
sum_local_overload(float _Complex val, float _Complex *rval, float _Complex *xteam_mem,
                   unsigned int *td_ptr, const float _Complex initval) {
  __kmpc_xteamr_cf_16x64(val, rval, xteam_mem, td_ptr, __kmpc_rfun_sum_cf,
                        __kmpc_rfun_sum_lds_cf, initval);
}
void __attribute__((flatten, always_inline))
sum_local_overload(int val, int *rval, int *xteam_mem, unsigned int *td_ptr,
                   const int initval) {
  __kmpc_xteamr_i_16x64(val, rval, xteam_mem, td_ptr, __kmpc_rfun_sum_i,
                        __kmpc_rfun_sum_lds_i, initval);
}
void __attribute__((flatten, always_inline))
sum_local_overload(unsigned int val, unsigned int *rval,
                   unsigned int *xteam_mem, unsigned int *td_ptr,
                   const unsigned int initval) {
  __kmpc_xteamr_ui_16x64(val, rval, xteam_mem, td_ptr, __kmpc_rfun_sum_ui,
                         __kmpc_rfun_sum_lds_ui, initval);
}
void __attribute__((flatten, always_inline))
sum_local_overload(long val, long *rval, long *xteam_mem, unsigned int *td_ptr,
                   const long initval) {
  __kmpc_xteamr_l_16x64(val, rval, xteam_mem, td_ptr, __kmpc_rfun_sum_l,
                        __kmpc_rfun_sum_lds_l, initval);
}
void __attribute__((flatten, always_inline))
sum_local_overload(unsigned long val, unsigned long *rval,
                   unsigned long *xteam_mem, unsigned int *td_ptr,
                   const unsigned long initval) {
  __kmpc_xteamr_ul_16x64(val, rval, xteam_mem, td_ptr, __kmpc_rfun_sum_ul,
                         __kmpc_rfun_sum_lds_ul, initval);
}

// ---- Local overloads for testing max
void __attribute__((flatten, always_inline))
max_local_overload(double val, double *rval, double *xteam_mem,
                   unsigned int *td_ptr, const double initval) {
  __kmpc_xteamr_d_16x64(val, rval, xteam_mem, td_ptr, __kmpc_rfun_max_d,
                        __kmpc_rfun_max_lds_d, initval);
}
void __attribute__((flatten, always_inline))
max_local_overload(float val, float *rval, float *xteam_mem,
                   unsigned int *td_ptr, const float initval) {
  __kmpc_xteamr_f_16x64(val, rval, xteam_mem, td_ptr, __kmpc_rfun_max_f,
                        __kmpc_rfun_max_lds_f, initval);
}
void __attribute__((flatten, always_inline))
max_local_overload(int val, int *rval, int *xteam_mem, unsigned int *td_ptr,
                   const int initval) {
  __kmpc_xteamr_i_16x64(val, rval, xteam_mem, td_ptr, __kmpc_rfun_max_i,
                        __kmpc_rfun_max_lds_i, initval);
}
void __attribute__((flatten, always_inline))
max_local_overload(unsigned int val, unsigned int *rval,
                   unsigned int *xteam_mem, unsigned int *td_ptr,
                   const unsigned int initval) {
  __kmpc_xteamr_ui_16x64(val, rval, xteam_mem, td_ptr, __kmpc_rfun_max_ui,
                         __kmpc_rfun_max_lds_ui, initval);
}
void __attribute__((flatten, always_inline))
max_local_overload(long val, long *rval, long *xteam_mem, unsigned int *td_ptr,
                   const long initval) {
  __kmpc_xteamr_l_16x64(val, rval, xteam_mem, td_ptr, __kmpc_rfun_max_l,
                        __kmpc_rfun_max_lds_l, initval);
}

void __attribute__((flatten, always_inline))
max_local_overload(unsigned long val, unsigned long *rval,
                   unsigned long *xteam_mem, unsigned int *td_ptr,
                   const unsigned long initval) {
  __kmpc_xteamr_ul_16x64(val, rval, xteam_mem, td_ptr, __kmpc_rfun_max_ul,
                         __kmpc_rfun_max_lds_ul, initval);
}

// ---- Local overloads for testing min
void __attribute__((flatten, always_inline))
min_local_overload(double val, double *rval, double *xteam_mem,
                   unsigned int *td_ptr, const double initval) {
  __kmpc_xteamr_d_16x64(val, rval, xteam_mem, td_ptr, __kmpc_rfun_min_d,
                        __kmpc_rfun_min_lds_d, initval);
}
void __attribute__((flatten, always_inline))
min_local_overload(float val, float *rval, float *xteam_mem,
                   unsigned int *td_ptr, const float initval) {
  __kmpc_xteamr_f_16x64(val, rval, xteam_mem, td_ptr, __kmpc_rfun_min_f,
                        __kmpc_rfun_min_lds_f, initval);
}
void __attribute__((flatten, always_inline))
min_local_overload(int val, int *rval, int *xteam_mem, unsigned int *td_ptr,
                   const int initval) {
  __kmpc_xteamr_i_16x64(val, rval, xteam_mem, td_ptr, __kmpc_rfun_min_i,
                        __kmpc_rfun_min_lds_i, initval);
}
void __attribute__((flatten, always_inline))
min_local_overload(unsigned int val, unsigned int *rval,
                   unsigned int *xteam_mem, unsigned int *td_ptr,
                   const unsigned int initval) {
  __kmpc_xteamr_ui_16x64(val, rval, xteam_mem, td_ptr, __kmpc_rfun_min_ui,
                         __kmpc_rfun_min_lds_ui, initval);
}
void __attribute__((flatten, always_inline))
min_local_overload(long val, long *rval, long *xteam_mem, unsigned int *td_ptr,
                   const long initval) {
  __kmpc_xteamr_l_16x64(val, rval, xteam_mem, td_ptr, __kmpc_rfun_min_l,
                        __kmpc_rfun_min_lds_l, initval);
}
void __attribute__((flatten, always_inline))
min_local_overload(unsigned long val, unsigned long *rval,
                   unsigned long *xteam_mem, unsigned int *td_ptr,
                   const unsigned long initval) {
  __kmpc_xteamr_ul_16x64(val, rval, xteam_mem, td_ptr, __kmpc_rfun_min_ul,
                         __kmpc_rfun_min_lds_ul, initval);
}

template <typename T> T omp_dot(T *a, T *b, int array_size) {
  T sum = 0.0;
#pragma omp target teams distribute parallel for map(tofrom: sum) reduction(+:sum)
  for (int i = 0; i < array_size; i++)
    sum += a[i] * b[i];
  return sum;
}

template <typename T> T omp_max(T *c, int array_size) {
  T maxval = std::numeric_limits<T>::lowest();
#pragma omp target teams distribute parallel for map(tofrom:maxval)            \
    reduction(max:maxval)
  for (int i = 0; i < array_size; i++)
    maxval = (c[i] > maxval) ? c[i] : maxval;
  return maxval;
}

template <typename T> T omp_min(T *c, int array_size) {
  T minval = std::numeric_limits<T>::max();
#pragma omp target teams distribute parallel for map(tofrom:minval)            \
    reduction(min:minval)
  for (int i = 0; i < array_size; i++) {
    minval = (c[i] < minval) ? c[i] : minval;
  }
  return minval;
}

#define _INNER_LOOP                                                            \
  for (int i = ((k * LOOP_STRIDE) + LOOP_START); i < LOOP_SIZE;                \
       i += (LOOP_TEAMS * _XTEAM_NUM_THREADS * LOOP_STRIDE))

template <typename T> T sim_dot(T *a, T *b, int array_size) {
  T sum = T(0);
  int devid = 0;
  static uint32_t *teams_done_ptr0 = nullptr;
  static uint32_t *d_teams_done_ptr0;
  static T *d_team_vals0;
  static uint32_t team_procs0;
  if (!teams_done_ptr0) {
    // One-time alloc device array for each teams's reduction value.
    team_procs0 = ompx_get_team_procs(devid);
    d_team_vals0 = (T *)omp_target_alloc(sizeof(T) * team_procs0, devid);
    // Allocate and copy the zero-initialized teams_done counter one time
    // because it atomically resets when last team increments it.
    teams_done_ptr0 = (uint32_t *)malloc(sizeof(uint32_t));
    *teams_done_ptr0 = 0;
    d_teams_done_ptr0 = (uint32_t *)omp_target_alloc(sizeof(uint32_t), devid);
    omp_target_memcpy(d_teams_done_ptr0, teams_done_ptr0, sizeof(uint32_t), 0,
                      0, devid, omp_get_initial_device());
  }
  // Making the array_size 64 bits avoids a data_submit and data_retrieve
  const uint64_t LOOP_TEAMS = team_procs0;
  const uint64_t LOOP_SIZE = (uint64_t)array_size;
#pragma omp target teams distribute parallel for num_teams(LOOP_TEAMS)         \
    num_threads(_XTEAM_NUM_THREADS) map(tofrom:sum)                            \
        is_device_ptr(d_team_vals0, d_teams_done_ptr0)
  for (unsigned int k = 0; k < (LOOP_TEAMS * _XTEAM_NUM_THREADS); k++) {
    T val0 = T(0);
    constexpr int LOOP_START = 0;
    constexpr uint32_t LOOP_STRIDE = 1;
    _INNER_LOOP  { val0 += a[i] * b[i]; }
    sum_local_overload(val0, &sum, d_team_vals0, d_teams_done_ptr0, T(0));
  }
  return sum;
}

template <typename T> T sim_max(T *c, int array_size) {
  int devid = 0;
  const T minval = std::numeric_limits<T>::lowest();
  T retval = minval;
  static uint32_t *teams_done_ptr1 = nullptr;
  static uint32_t *d_teams_done_ptr1;
  static T *d_team_vals1;
  static uint32_t team_procs1;
  if (!teams_done_ptr1) {
    // One-time alloc device array for each teams's reduction value.
    team_procs1 = ompx_get_team_procs(devid);
    d_team_vals1 = (T *)omp_target_alloc(sizeof(T) * team_procs1, devid);
    // Allocate and copy the zero-initialized teams_done counter one time
    // because it atomically resets when last team increments it.
    // Clang can create a global initialized to 0 and remove the alloc and copy
    teams_done_ptr1 = (uint32_t *)malloc(sizeof(uint32_t));
    *teams_done_ptr1 = 0;
    d_teams_done_ptr1 = (uint32_t *)omp_target_alloc(sizeof(uint32_t), devid);
    omp_target_memcpy(d_teams_done_ptr1, teams_done_ptr1, sizeof(uint32_t), 0,
                      0, devid, omp_get_initial_device());
  }
  // Making the array_size 64 bits somehow avoids a data_submit and
  // data_retrieve.?
  const uint64_t LOOP_TEAMS = team_procs1;
  const uint64_t LOOP_SIZE = (uint64_t)array_size;
  const T LOOP_INITVAL = minval;
#pragma omp target teams distribute parallel for num_teams(LOOP_TEAMS)         \
    num_threads(_XTEAM_NUM_THREADS) map(tofrom                                 \
                                        : retval)                              \
        is_device_ptr(d_team_vals1, d_teams_done_ptr1)
  for (unsigned int k = 0; k < (LOOP_TEAMS * _XTEAM_NUM_THREADS); k++) {
    T val1 = LOOP_INITVAL;
    const int LOOP_START = 0;
    const uint32_t LOOP_STRIDE = 1;
    _INNER_LOOP { val1 = (c[i] > val1) ? c[i] : val1; }
    // we do not have device numeric_limits so use host numeric_limits
    // clang codegen should embed a constant for each datatype to pass to xteamr
    max_local_overload(val1, &retval, d_team_vals1, d_teams_done_ptr1,
                       LOOP_INITVAL);
  }
  return retval;
}

template <typename T> T sim_min(T *c, int array_size) {
  int devid = 0;
  const T maxval = std::numeric_limits<T>::max();
  T retval = maxval;
  static uint32_t *teams_done_ptr2;
  static uint32_t *d_teams_done_ptr2;
  static T *d_team_vals2;
  static uint32_t team_procs2;
  if (!teams_done_ptr2) {
    // One-time alloc device array for each teams's reduction value.
    team_procs2 = ompx_get_team_procs(devid);
    d_team_vals2 = (T *)omp_target_alloc(sizeof(T) * team_procs2, devid);
    // Allocate and copy the zero-initialized teams_done counter one time
    // because it atomically resets when last team increments it.
    // Clang can create a global initialized to 0 and remove the alloc and copy
    teams_done_ptr2 = (uint32_t *)malloc(sizeof(uint32_t));
    *teams_done_ptr2 = 0;
    d_teams_done_ptr2 = (uint32_t *)omp_target_alloc(sizeof(uint32_t), devid);
    omp_target_memcpy(d_teams_done_ptr2, teams_done_ptr2, sizeof(uint32_t), 0,
                      0, devid, omp_get_initial_device());
  }
  // Making the array_size 64 bits avoids a data_submit and data_retrieve.
  const uint64_t LOOP_TEAMS = team_procs2;
  const uint64_t LOOP_SIZE = (uint64_t)array_size;
  const T LOOP_INITVAL = maxval;
#pragma omp target teams distribute parallel for num_teams(LOOP_TEAMS)         \
    num_threads(_XTEAM_NUM_THREADS) map(tofrom:retval)                         \
        is_device_ptr(d_team_vals2, d_teams_done_ptr2)
  for (unsigned int k = 0; k < (LOOP_TEAMS * _XTEAM_NUM_THREADS); k++) {
    T val2 = LOOP_INITVAL;
    const int LOOP_START = 0;
    const uint32_t LOOP_STRIDE = 1;
    _INNER_LOOP { val2 = (c[i] < val2) ? c[i] : val2; }
    min_local_overload(val2, &retval, d_team_vals2, d_teams_done_ptr2,
                       LOOP_INITVAL);
  }
  return retval;
}

template <typename T, bool DATA_TYPE_IS_INT>
void _check_val(T computed_val, T gold_val, const char *msg) {
  double ETOL = 0.0000001;
  if (DATA_TYPE_IS_INT) {
    if (computed_val != gold_val) {
      std::cerr << msg << " FAIL "
                << "Integar Value was " << computed_val << " but should be "
                << gold_val << std::endl;
      smoke_rc = 1;
    }
  } else {
    double dcomputed_val = (double)computed_val;
    double dvalgold = (double)gold_val;
    double ompErrSum = abs((dcomputed_val - dvalgold) / dvalgold);
    if (ompErrSum > ETOL) {
      std::cerr << msg << " FAIL " << ompErrSum << " tol:" << ETOL << std::endl
                << std::setprecision(15) << "Value was " << computed_val
                << " but should be " << gold_val << std::endl;
      smoke_rc = 1;
    }
  }
}

template <typename T, bool DATA_TYPE_IS_INT>
void run_kernels(const int ARRAY_SIZE) {
  int array_size = ARRAY_SIZE;
  T *a = (T *)aligned_alloc(ALIGNMENT, sizeof(T) * array_size);
  T *b = (T *)aligned_alloc(ALIGNMENT, sizeof(T) * array_size);
  T *c = (T *)aligned_alloc(ALIGNMENT, sizeof(T) * array_size);
#pragma omp target enter data map(alloc                                        \
                                  : a [0:array_size], b [0:array_size],        \
                                    c [0:array_size])
#pragma omp target teams distribute parallel for
  for (int i = 0; i < array_size; i++) {
    a[i] = 2;
    b[i] = 3;
    c[i] = (i + 1);
  }

  std::cout << "Running kernels " << num_times << " times" << std::endl;
  std::cout << "Ignoring timing of first " << ignore_num_times << "  runs "
            << std::endl;

  double ETOL = 0.0000001;
  if (DATA_TYPE_IS_INT) {
    std::cout << "Integer Size: " << sizeof(T) << std::endl;
  } else {
    if (sizeof(T) == sizeof(float))
      std::cout << "Precision: float" << std::endl;
    else
      std::cout << "Precision: double" << std::endl;
  }

  std::cout << "Array elements: " << ARRAY_SIZE << std::endl;
  std::cout << "Array size:     " << ((ARRAY_SIZE * sizeof(T)) / 1024 * 1024)
            << " MB" << std::endl;

  T goldDot = (T)6 * (T)ARRAY_SIZE;
  T goldMax = (T)ARRAY_SIZE;
  T goldMin = (T)1;

  double goldDot_d = (double)goldDot;
  double goldMax_d = (double)goldMax;
  double goldMin_d = (double)goldMin;

  // List of times
  std::vector<std::vector<double>> timings(6);

  // Declare timers
  std::chrono::high_resolution_clock::time_point  t1, t2;

  // Main loop
  for (unsigned int k = 0; k < num_times; k++) {
    t1 = std::chrono::high_resolution_clock::now();
    T omp_sum = omp_dot<T>(a, b, array_size);
    t2 = std::chrono::high_resolution_clock::now();
    timings[0].push_back(
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count());
    _check_val<T, DATA_TYPE_IS_INT>(omp_sum, goldDot, "omp_dot");

    t1 = std::chrono::high_resolution_clock::now();
    T sim_sum = sim_dot<T>(a, b, array_size);
    t2 = std::chrono::high_resolution_clock::now();
    timings[1].push_back(
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count());
    _check_val<T, DATA_TYPE_IS_INT>(sim_sum, goldDot, "sim_dot");

    t1 = std::chrono::high_resolution_clock::now();
    T omp_max_val = omp_max<T>(c, array_size);
    t2 = std::chrono::high_resolution_clock::now();
    timings[2].push_back(
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count());
    _check_val<T, DATA_TYPE_IS_INT>(omp_max_val, goldMax, "omp_max");

    t1 = std::chrono::high_resolution_clock::now();
    T sim_max_val = sim_max<T>(c, array_size);
    t2 = std::chrono::high_resolution_clock::now();
    timings[3].push_back(
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count());
    _check_val<T, DATA_TYPE_IS_INT>(sim_max_val, goldMax, "sim_max");

    t1 = std::chrono::high_resolution_clock::now();
    T omp_min_val = omp_min<T>(c, array_size);
    t2 = std::chrono::high_resolution_clock::now();
    timings[4].push_back(
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count());
    _check_val<T, DATA_TYPE_IS_INT>(omp_min_val, goldMin, "omp_min");

    t1 = std::chrono::high_resolution_clock::now();
    T sim_min_val = sim_min<T>(c, array_size);
    t2 = std::chrono::high_resolution_clock::now();
    timings[5].push_back(
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count());
    _check_val<T, DATA_TYPE_IS_INT>(sim_min_val, goldMin, "sim_min");

  } // end for (unsigned int k = 0; k < num_times; k++)

  // Display timing results
  std::cout << std::left << std::setw(12) << "Function" << std::left
            << std::setw(12) << "Best-MB/sec" << std::left << std::setw(12)
            << " Min (sec)" << std::left << std::setw(12) << "   Max"
            << std::left << std::setw(12) << "Average" << std::left
            << std::setw(12) << "Avg-MB/sec" << std::endl;

  std::cout << std::fixed;

  std::string labels[6] = {"ompdot", "simdot", "ompmax",
                           "simmax", "ompmin", "simmin"};
  size_t sizes[6] = {2 * sizeof(T) * ARRAY_SIZE, 2 * sizeof(T) * ARRAY_SIZE,
                     1 * sizeof(T) * ARRAY_SIZE, 1 * sizeof(T) * ARRAY_SIZE,
                     1 * sizeof(T) * ARRAY_SIZE, 1 * sizeof(T) * ARRAY_SIZE};

  for (int i = 0; i < 6; i++) {
    // Get min/max; ignore the first couple results
    auto minmax = std::minmax_element(timings[i].begin() + ignore_num_times,
                                      timings[i].end());

    // Calculate average; ignore ignore_num_times
    double average = std::accumulate(timings[i].begin() + ignore_num_times,
                                     timings[i].end(), 0.0) /
                     (double)(num_times - ignore_num_times);

    printf("  %s       %8.0f   %8.6f  %8.6f   %8.6f    %8.0f\n",
           labels[i].c_str(), 1.0E-6 * sizes[i] / (*minmax.first),
           (double)*minmax.first, (double)*minmax.second, (double)average,
           1.0E-6 * sizes[i] / (average));
  }
#pragma omp target exit data map(release                                       \
                                 : a [0:array_size], b [0:array_size],         \
                                   c [0:array_size])
  free(a);
  free(b);
  free(c);
}

/// ======================== START COMPLEX ROUTINES ===========================
//
template <typename TC, typename T>
void _check_val_complex(TC computed_val_complex, TC gold_val_complex, const char *msg) {
  T  gold_val_r = __real__(gold_val_complex);
  T  computed_val_r = __real__(computed_val_complex);
  T  gold_val_i = __imag__(gold_val_complex);
  T  computed_val_i = __imag__(computed_val_complex);
  double ETOL = 0.0000001;
  double computed_val_r_d = (double)computed_val_r;
  double valgold_r_d = (double)gold_val_r;
  double ompErrSum_r = abs((computed_val_r_d - valgold_r_d) / valgold_r_d);
  double computed_val_i_d = (double)computed_val_i;
  double valgold_i_d = (double)gold_val_i;
  double ompErrSum_i = abs((computed_val_i_d - valgold_i_d) / valgold_i_d);
    if ((ompErrSum_r > ETOL) || (ompErrSum_i > ETOL)) {
      std::cerr << msg << " FAIL " << ompErrSum_r << " tol:" << ETOL << std::endl
      << std::setprecision(15) << "Value was (" << computed_val_r << " + " << computed_val_i << " i )" << std::endl 
      << " but should be (" << gold_val_r << " + " <<  gold_val_i  << "i) " << std::endl;
      smoke_rc = 1;
    }
}

template <typename TC>
TC omp_dot_complex(TC *a, TC *b, int array_size) {
  TC dot ; __real__(dot) = 0.0; __imag__(dot) = 0.0;
#pragma omp target teams distribute parallel for map(tofrom: dot) reduction(+:dot)
  for (int i = 0; i < array_size; i++)
    dot += a[i] * b[i];
  return dot;
}

template <typename TC>
TC sim_dot_complex(TC *a, TC *b, int array_size) {
  TC zero_c ; __real__(zero_c) = 0.0; __imag__(zero_c) = 0.0;
  TC  sum = zero_c;
  int devid = 0;
  static uint32_t *teams_done_ptr00 = nullptr;
  static uint32_t *d_teams_done_ptr00;
  static TC *d_team_vals00;
  static uint32_t team_procs00;
  if (!teams_done_ptr00) {
    // One-time alloc device array for each teams's reduction value.
    team_procs00 = ompx_get_team_procs(devid);
    d_team_vals00 = (TC *)omp_target_alloc(sizeof(TC) * team_procs00, devid);
    // Allocate and copy the zero-initialized teams_done counter one time
    // because it atomically resets when last team increments it.
    teams_done_ptr00 = (uint32_t *)malloc(sizeof(uint32_t));
    *teams_done_ptr00 = 0;
    d_teams_done_ptr00 = (uint32_t *)omp_target_alloc(sizeof(uint32_t), devid);
    omp_target_memcpy(d_teams_done_ptr00, teams_done_ptr00, sizeof(uint32_t), 0,
                      0, devid, omp_get_initial_device());
  }
  // Making the array_size 64 bits avoids a data_submit and data_retrieve
  const uint64_t LOOP_TEAMS = team_procs00;
  const uint64_t LOOP_SIZE = (uint64_t)array_size;
#pragma omp target teams distribute parallel for num_teams(LOOP_TEAMS)         \
    num_threads(_XTEAM_NUM_THREADS) map(tofrom:sum)                            \
        is_device_ptr(d_team_vals00, d_teams_done_ptr00)
  for (unsigned int k = 0; k < (LOOP_TEAMS * _XTEAM_NUM_THREADS); k++) {
    TC val00 = zero_c;
    constexpr int LOOP_START = 0;
    constexpr uint32_t LOOP_STRIDE = 1;
    _INNER_LOOP { val00 += a[i] * b[i]; }
    sum_local_overload(val00, &sum, d_team_vals00, d_teams_done_ptr00, zero_c);
  }
  return sum;
}

template <typename TC, typename T>
void run_kernels_complex(const int array_size) {

  TC *a = (TC *)aligned_alloc(ALIGNMENT, sizeof(TC) * array_size);
  TC *b = (TC *)aligned_alloc(ALIGNMENT, sizeof(TC) * array_size);

  #pragma omp target enter data map(alloc:a [0:array_size], b [0:array_size])
  TC startA; __real__(startA) = 1.0; __imag__(startA) = 1.0;
  TC startB; __real__(startB) = 1.0; __imag__(startB) = -1.0;

#pragma omp target teams distribute parallel for
  for (int i = 0; i < array_size; i++) {
    a[i] = startA;
    b[i] = startB;
    // a[i] * b[i] = 2 + 0i
  }

  std::cout << "Running kernels " << num_times << " times" << std::endl;
  std::cout << "Ignoring timing of first " << ignore_num_times << "  runs "
            << std::endl;

  double ETOL = 0.0000001;
    if (sizeof(TC) == sizeof(float _Complex))
      std::cout << "Precision: float _Complex" << std::endl;
    else
      std::cout << "Precision: double _Complex" << std::endl;

  std::cout << "Array elements: " << array_size << std::endl;
  std::cout << "Array size:     " << ((array_size* sizeof(TC)) / 1024 * 1024)
            << " MB" << std::endl;

  T goldDotr = T(2) * (T)array_size;
  T goldDoti = T(0);

  TC goldDot ; __real__(goldDot) = goldDotr; __imag__(goldDot) = goldDoti;

  // List of times
  std::vector<std::vector<double>> timings(2);

  // Declare timers
  std::chrono::high_resolution_clock::time_point t1, t2;

  // Main loop
  for (unsigned int k = 0; k < num_times; k++) {
    t1 = std::chrono::high_resolution_clock::now();
    TC omp_sum = omp_dot_complex<TC>(a, b, array_size);
    t2 = std::chrono::high_resolution_clock::now();
    timings[0].push_back(
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count());
    _check_val_complex<TC,T>(omp_sum, goldDot, "omp_dot");

    t1 = std::chrono::high_resolution_clock::now();
    TC sim_sum = sim_dot_complex<TC>(a, b, array_size);
    t2 = std::chrono::high_resolution_clock::now();
    timings[1].push_back(
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count());
    _check_val_complex<TC,T>(sim_sum, goldDot, "sim_dot");


  } // end for (unsigned int k = 0; k < num_times; k++)

  // Display timing results
  std::cout << std::left << std::setw(12) << "Function" << std::left
            << std::setw(12) << "Best-MB/sec" << std::left << std::setw(12)
            << " Min (sec)" << std::left << std::setw(12) << "   Max"
            << std::left << std::setw(12) << "Average" << std::left
            << std::setw(12) << "Avg-MB/sec" << std::endl;

  std::cout << std::fixed;

  std::string labels[2] = {"ompdot", "simdot"};
  size_t sizes[2] = {2 * sizeof(TC) * array_size, 2 * sizeof(TC) * array_size};

  for (int i = 0; i < 2; i++) {
    // Get min/max; ignore the first couple results
    auto minmax = std::minmax_element(timings[i].begin() + ignore_num_times,
                                      timings[i].end());

    // Calculate average; ignore ignore_num_times
    double average = std::accumulate(timings[i].begin() + ignore_num_times,
                                     timings[i].end(), 0.0) /
                     (double)(num_times - ignore_num_times);

    printf("  %s       %8.0f   %8.6f  %8.6f   %8.6f    %8.0f\n",
           labels[i].c_str(), 1.0E-6 * sizes[i] / (*minmax.first),
           (double)*minmax.first, (double)*minmax.second, (double)average,
           1.0E-6 * sizes[i] / (average));
  }
  #pragma omp target exit data map(release : a [0:array_size], b [0:array_size])
  free(a);
  free(b);
}
