#include <omp.h>
#include <cstdlib>  // For aligned_alloc
#include <vector>

#ifndef ALIGNMENT
#define ALIGNMENT (2*1024*1024) // 2MB
#endif

#include <iostream>
#include <stdexcept>

#include <vector>
#include <numeric>
#include <cmath>
#include <limits>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cstring>

#include "DeviceDeclares.h"
#include "HostDefs.h"

#define _XTEAM_NUM_THREADS 1024

const int ARRAY_SIZE = 33554432;
//const int ARRAY_SIZE = 33;
unsigned int num_times = 12;
unsigned int ignore_num_times = 2;
unsigned int smoke_rc = 0;

template <typename T, bool >
void run_kernels(const int ARRAY_SIZE);

int main(int argc, char *argv[]) {
  std::cout << std::endl << "TEST DOUBLE" << std::endl;
  run_kernels<double,false>(ARRAY_SIZE);
  std::cout << std::endl << "TEST FLOAT"<< std::endl;
  run_kernels<float,false>(ARRAY_SIZE);
  std::cout << std::endl << "TEST INT" << std::endl;
  run_kernels<int,true>(ARRAY_SIZE);
  std::cout << std::endl << "TEST UNSIGNED INT" << std::endl;
  run_kernels<unsigned,true>(ARRAY_SIZE);
  std::cout << std::endl << "TEST LONG" << std::endl;
  run_kernels<long,true>(ARRAY_SIZE);
  std::cout << std::endl << "TEST UNSIGNED LONG" << std::endl;
  run_kernels<unsigned long,true>(ARRAY_SIZE);
  return smoke_rc;
}

// ---- Local overloads for testing sum
void __attribute__((flatten, always_inline)) sum_local_overload(double val, double* rval,
	       	double* xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_d_16x64(val,rval,xteam_mem,td_ptr, __kmpc_rfun_sum_d,__kmpc_rfun_sum_lds_d, 0.0);
}
void __attribute__((flatten, always_inline)) sum_local_overload(float val, float* rval,
	       	float* xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_f_16x64(val,rval,xteam_mem,td_ptr, __kmpc_rfun_sum_f,__kmpc_rfun_sum_lds_f, 0.0);
}
void __attribute__((flatten, always_inline)) sum_local_overload(int val, int* rval,
	       	int* xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_i_16x64(val,rval,xteam_mem,td_ptr, __kmpc_rfun_sum_i,__kmpc_rfun_sum_lds_i, 0);
}
void __attribute__((flatten, always_inline)) sum_local_overload(unsigned int val, unsigned int* rval,
	       	unsigned int * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_ui_16x64(val,rval,xteam_mem,td_ptr, __kmpc_rfun_sum_ui,__kmpc_rfun_sum_lds_ui, 0);
}
void __attribute__((flatten, always_inline)) sum_local_overload(long val, long* rval,
	       	long * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_l_16x64(val,rval, xteam_mem,td_ptr, __kmpc_rfun_sum_l,__kmpc_rfun_sum_lds_l, 0);
}
void __attribute__((flatten, always_inline)) sum_local_overload(unsigned long val, unsigned long * rval,
	       	unsigned long * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_ul_16x64(val,rval, xteam_mem,td_ptr, __kmpc_rfun_sum_ul,__kmpc_rfun_sum_lds_ul, 0);
}

#define __XTEAM_MAX_FLOAT (__builtin_inff())
#define __XTEAM_LOW_FLOAT -__XTEAM_MAX_FLOAT
#define __XTEAM_MAX_DOUBLE (__builtin_huge_val())
#define __XTEAM_LOW_DOUBLE -__XTEAM_MAX_DOUBLE
#define __XTEAM_MAX_INT32 2147483647
#define __XTEAM_LOW_INT32 (-__XTEAM_MAX_INT32 - 1)
#define __XTEAM_MAX_UINT32 4294967295
#define __XTEAM_LOW_UINT32 0
#define __XTEAM_MAX_INT64 9223372036854775807
#define __XTEAM_LOW_INT64 (-__XTEAM_MAX_INT64 - 1)
#define __XTEAM_MAX_UINT64 0xffffffffffffffff
#define __XTEAM_LOW_UINT64 0

// ---- Local overloads for testing max
void __attribute__((flatten, always_inline)) max_local_overload(double val, double* rval,
	double * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_d_16x64(val,rval,xteam_mem, td_ptr,
		  __kmpc_rfun_max_d,__kmpc_rfun_max_lds_d, __XTEAM_LOW_DOUBLE);
}
void __attribute__((flatten, always_inline)) max_local_overload(float val, float* rval,
	float * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_f_16x64(val,rval,xteam_mem, td_ptr,
		  __kmpc_rfun_max_f,__kmpc_rfun_max_lds_f, __XTEAM_LOW_FLOAT);
}
void __attribute__((flatten, always_inline)) max_local_overload(int val, int* rval,
	int * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_i_16x64(val,rval,xteam_mem, td_ptr,
		  __kmpc_rfun_max_i,__kmpc_rfun_max_lds_i, __XTEAM_LOW_INT32);
}
void __attribute__((flatten, always_inline)) max_local_overload(unsigned int val, unsigned int * rval,
	unsigned int * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_ui_16x64(val,rval,xteam_mem, td_ptr,
		  __kmpc_rfun_max_ui,__kmpc_rfun_max_lds_ui, 0u);
}
void __attribute__((flatten, always_inline)) max_local_overload(long val, long * rval,
	long * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_l_16x64(val,rval,xteam_mem, td_ptr,
		  __kmpc_rfun_max_l,__kmpc_rfun_max_lds_l, __XTEAM_LOW_INT64);
}
void __attribute__((flatten, always_inline)) max_local_overload(unsigned long val, unsigned long * rval,
	unsigned long * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_ul_16x64(val,rval,xteam_mem, td_ptr,
		  __kmpc_rfun_max_ul,__kmpc_rfun_max_lds_ul, 0ul);
}

// ---- Local overloads for testing min
void __attribute__((flatten, always_inline)) min_local_overload(double val, double* rval,
	double * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_d_16x64(val,rval,xteam_mem, td_ptr,
		  __kmpc_rfun_min_d,__kmpc_rfun_min_lds_d, __XTEAM_MAX_DOUBLE);
}
void __attribute__((flatten, always_inline)) min_local_overload(float val, float* rval,
	float * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_f_16x64(val,rval,xteam_mem, td_ptr,
		  __kmpc_rfun_min_f,__kmpc_rfun_min_lds_f, __XTEAM_MAX_FLOAT);
}
void __attribute__((flatten, always_inline)) min_local_overload(int val, int* rval,
	int * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_i_16x64(val,rval,xteam_mem, td_ptr,
		  __kmpc_rfun_min_i,__kmpc_rfun_min_lds_i, __XTEAM_MAX_INT32);
}
void __attribute__((flatten, always_inline)) min_local_overload(unsigned int val, unsigned int * rval,
	unsigned int * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_ui_16x64(val,rval,xteam_mem, td_ptr,
		  __kmpc_rfun_min_ui,__kmpc_rfun_min_lds_ui, __XTEAM_MAX_UINT32);
}
void __attribute__((flatten, always_inline)) min_local_overload(long val, long * rval,
	long * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_l_16x64(val,rval,xteam_mem, td_ptr,
		  __kmpc_rfun_min_l,__kmpc_rfun_min_lds_l, __XTEAM_MAX_INT64);
}
void __attribute__((flatten, always_inline)) min_local_overload(unsigned long val, unsigned long * rval,
	 unsigned long * xteam_mem,  unsigned int * td_ptr) {
  __kmpc_xteamr_ul_16x64(val,rval,xteam_mem, td_ptr,
		  __kmpc_rfun_min_ul,__kmpc_rfun_min_lds_ul, __XTEAM_MAX_UINT64);
}

    template <typename T>
    T omp_dot(T*a, T*b, int array_size) {
      T sum = 0.0;
      #pragma omp target teams distribute parallel for map(tofrom: sum) reduction(+:sum)
      for (int i = 0; i < array_size; i++)
        sum += a[i] * b[i];
      return sum;
    }

    template <typename T>
    T omp_max(T*c, int array_size) {
      T maxval = std::numeric_limits<T>::lowest();
      #pragma omp target teams distribute parallel for map(tofrom: maxval) reduction(max:maxval)
      for (int i = 0; i < array_size; i++)
	maxval = (c[i] > maxval) ? c[i] : maxval;
      return maxval;
    }

    template <typename T>
    T omp_min(T*c, int array_size) {
      T minval = std::numeric_limits<T>::max();
      #pragma omp target teams distribute parallel for map(tofrom: minval) reduction(min:minval)
      for (int i = 0; i < array_size; i++) {
	minval = (c[i] < minval) ? c[i] : minval;
      }
      return minval;
    }

    template <typename T>
    T sim_dot(T*a, T*b, int array_size) {
      T sum = T(0);
      int devid =  0;
      static uint32_t * teams_done_ptr0 = nullptr;
      static uint32_t * d_teams_done_ptr0;
      static T* d_team_vals0;
      static uint32_t team_procs0;
      if ( !teams_done_ptr0 ) {
         // One-time alloc device array for each teams's reduction value.
         team_procs0 = ompx_get_team_procs(devid);
         d_team_vals0  = (T *) omp_target_alloc(sizeof(T) * team_procs0, devid);
	 // Allocate and copy the zero-initialized teams_done counter one time
	 // because it atomically resets when last team increments it.
         teams_done_ptr0 = (uint32_t *)  malloc(sizeof(uint32_t));
         *teams_done_ptr0 = 0;
         d_teams_done_ptr0 = (uint32_t *) omp_target_alloc(sizeof(uint32_t),devid);
         omp_target_memcpy(d_teams_done_ptr0, teams_done_ptr0, 
	     sizeof(uint32_t), 0, 0, devid, omp_get_initial_device());
      }
      // Making the array_size 64 bits avoids a data_submit and data_retrieve 
      const uint64_t team_procs = team_procs0;
      const uint64_t as64 = (uint64_t) array_size;
      #pragma omp target teams distribute parallel for \
         num_teams(team_procs) num_threads(_XTEAM_NUM_THREADS) \
         map(tofrom:sum) is_device_ptr(d_team_vals0,d_teams_done_ptr0)
      for (unsigned int k=0; k<(team_procs*_XTEAM_NUM_THREADS); k++) {
        T val0 = T(0);
        for (unsigned int i = k; i < as64 ; i += team_procs*_XTEAM_NUM_THREADS) {
          val0 += a[i] * b[i];
	}
        sum_local_overload(val0, &sum, d_team_vals0, d_teams_done_ptr0);
      }
      return sum;
    }

    template <typename T>
    T sim_max(T*c, int array_size) {
      int devid =  0;
      T minval = std::numeric_limits<T>::lowest();
      T retval = minval;
      static uint32_t * teams_done_ptr1 = nullptr;
      static uint32_t * d_teams_done_ptr1;
      static T* d_team_vals1;
      static uint32_t team_procs1;
      if ( !teams_done_ptr1 ) {
         // One-time alloc device array for each teams's reduction value.
         team_procs1 = ompx_get_team_procs(devid);
         d_team_vals1  = (T *) omp_target_alloc(sizeof(T) * team_procs1, devid);
	 // Allocate and copy the zero-initialized teams_done counter one time
	 // because it atomically resets when last team increments it.
         teams_done_ptr1 = (uint32_t *)  malloc(sizeof(uint32_t));
         *teams_done_ptr1 = 0;
         d_teams_done_ptr1 = (uint32_t *) omp_target_alloc(sizeof(uint32_t),devid);
         omp_target_memcpy(d_teams_done_ptr1, teams_done_ptr1, 
	     sizeof(uint32_t), 0, 0, devid, omp_get_initial_device());
      }
      // Making the array_size 64 bits somehow avoids a data_submit and data_retrieve.?
      const uint64_t team_procs = team_procs1;
      const uint64_t as64 = (uint64_t) array_size;
      #pragma omp target teams distribute parallel for \
         num_teams(team_procs) num_threads(_XTEAM_NUM_THREADS) \
         map(tofrom:retval) is_device_ptr(d_team_vals1,d_teams_done_ptr1)
      for (unsigned int k=0; k<(team_procs*_XTEAM_NUM_THREADS); k++) {
        T val1 = retval;
        for (unsigned int i = k; i < as64 ; i += team_procs*_XTEAM_NUM_THREADS){
	  val1 = (c[i] > val1) ? c[i] : val1;
	}
        max_local_overload(val1, &retval, d_team_vals1, d_teams_done_ptr1);
      }
      return retval;
    }

    template <typename T>
    T sim_min(T*c, int array_size) {
      int devid =  0;
      T maxval = std::numeric_limits<T>::max();
      T retval = maxval;
      static uint32_t * teams_done_ptr2;
      static uint32_t * d_teams_done_ptr2;
      static T* d_team_vals2;
      static uint32_t team_procs2;
      if ( !teams_done_ptr2 ) {
         // One-time alloc device array for each teams's reduction value.
         team_procs2 = ompx_get_team_procs(devid);
         d_team_vals2  = (T *) omp_target_alloc(sizeof(T) * team_procs2, devid);
	 // Allocate and copy the zero-initialized teams_done counter one time
	 // because it atomically resets when last team increments it.
         teams_done_ptr2 = (uint32_t *)  malloc(sizeof(uint32_t));
         *teams_done_ptr2 = 0;
         d_teams_done_ptr2 = (uint32_t *) omp_target_alloc(sizeof(uint32_t),devid);
         omp_target_memcpy(d_teams_done_ptr2, teams_done_ptr2, 
	     sizeof(uint32_t), 0, 0, devid, omp_get_initial_device());
      }
      // Making the array_size 64 bits avoids a data_submit and data_retrieve.
      const uint64_t team_procs = team_procs2;
      const uint64_t as64 = (uint64_t) array_size;
      #pragma omp target teams distribute parallel for \
         num_teams(team_procs) num_threads(_XTEAM_NUM_THREADS) \
         map(tofrom:retval) is_device_ptr(d_team_vals2,d_teams_done_ptr2)
      for (unsigned int k=0; k<(team_procs*_XTEAM_NUM_THREADS); k++) {
        T val2 = retval;
        for (unsigned int i = k; i < as64 ; i += team_procs*_XTEAM_NUM_THREADS){
	  val2 = (c[i] < val2) ? c[i] : val2;
	}
        min_local_overload(val2, &retval, d_team_vals2, d_teams_done_ptr2);
      }
      return retval;
    }

template <typename T, bool DATA_TYPE_IS_INT>
void _check_val(T computed_val , T gold_val, const char*msg){
    double ETOL = 0.0000001;
    if (DATA_TYPE_IS_INT) {
      if (computed_val != gold_val) {
         std::cerr
        << msg << " FAIL "
        << "Integar Value was " << computed_val << " but should be " << gold_val
        << std::endl;
         smoke_rc = 1;
      }
    } else {
      double dcomputed_val = (double) computed_val;
      double dvalgold = (double) gold_val;
      double ompErrSum = abs((dcomputed_val - dvalgold)/dvalgold);
      if (ompErrSum > ETOL ) {
         std::cerr
        << msg << " FAIL " << ompErrSum << " tol:" << ETOL << std::endl << std::setprecision(15)
        << "Value was " << computed_val << " but should be " << gold_val
        << std::endl;
         smoke_rc = 1;
      }
    }
}

template <typename T, bool DATA_TYPE_IS_INT>
void run_kernels(const int ARRAY_SIZE) {
  int array_size = ARRAY_SIZE;
  T*a = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*array_size);
  T*b = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*array_size);
  T*c = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*array_size);
  #pragma omp target enter data map(alloc: a[0:array_size], b[0:array_size], c[0:array_size])
  #pragma omp target teams distribute parallel for
  for (int i = 0; i < array_size; i++) {
    a[i] = 2;
    b[i] = 3;
    c[i] = (i+1);
  }

  std::cout << "Running kernels " << num_times << " times" << std::endl;
  std::cout << "Ignoring timing of first "<< ignore_num_times << "  runs " << std::endl;

  double ETOL = 0.0000001;
  if (DATA_TYPE_IS_INT) { 
    std::cout << "Integer Size: " << sizeof(T)  << std::endl;
  } else {
    if (sizeof(T) == sizeof(float))
      std::cout << "Precision: float" << std::endl;
    else
      std::cout << "Precision: double" << std::endl;
  }

  std::cout << "Array elements: " << ARRAY_SIZE << std::endl ;
  std::cout << "Array size:     " << ((ARRAY_SIZE*sizeof(T)) / 1024*1024) << " MB" << std::endl;

  T goldDot = (T) 6  * (T) ARRAY_SIZE;
  T goldMax = (T) ARRAY_SIZE; 
  T goldMin = (T) 1;

  double goldDot_d = (double) goldDot;
  double goldMax_d = (double) goldMax;
  double goldMin_d = (double) goldMin;

  // List of times
  std::vector<std::vector<double>> timings(6);

  // Declare timers
  std::chrono::high_resolution_clock::time_point t0, t1, t2;

  // Main loop
  for (unsigned int k = 0; k < num_times; k++) {
    t1 = std::chrono::high_resolution_clock::now();
    T omp_sum = omp_dot<T>(a,b,array_size);
    t2 = std::chrono::high_resolution_clock::now();
    timings[0].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
    _check_val<T,DATA_TYPE_IS_INT>(omp_sum, goldDot, "omp_dot");

    t1 = std::chrono::high_resolution_clock::now();
    T sim_sum =  sim_dot<T>(a,b,array_size);
    t2 = std::chrono::high_resolution_clock::now();
    timings[1].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
    _check_val<T,DATA_TYPE_IS_INT>(sim_sum, goldDot, "sim_dot");

    t1 = std::chrono::high_resolution_clock::now();
    T omp_max_val = omp_max<T>(c,array_size);
    t2 = std::chrono::high_resolution_clock::now();
    timings[2].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
    _check_val<T,DATA_TYPE_IS_INT>(omp_max_val, goldMax, "sim_max");

    t1 = std::chrono::high_resolution_clock::now();
    T sim_max_val = sim_max<T>(c,array_size);
    t2 = std::chrono::high_resolution_clock::now();
    timings[3].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
    _check_val<T,DATA_TYPE_IS_INT>(sim_max_val, goldMax, "sim_max");

    t1 = std::chrono::high_resolution_clock::now();
    T omp_min_val = omp_min<T>(c,array_size);
    t2 = std::chrono::high_resolution_clock::now();
    timings[4].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
    _check_val<T,DATA_TYPE_IS_INT>(omp_min_val, goldMin, "omp_min");

    t1 = std::chrono::high_resolution_clock::now();
    T sim_min_val = sim_min<T>(c,array_size);
    t2 = std::chrono::high_resolution_clock::now();
    timings[5].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
    _check_val<T,DATA_TYPE_IS_INT>(sim_min_val, goldMin, "sim_min");
     
  }  // end for (unsigned int k = 0; k < num_times; k++)


  // Display timing results
  std::cout
    << std::left << std::setw(12) << "Function"
    << std::left << std::setw(12) << "Best-MB/sec"
    << std::left << std::setw(12) << " Min (sec)"
    << std::left << std::setw(12) << "   Max"
    << std::left << std::setw(12) << "Average" 
    << std::left << std::setw(12) << "Avg-MB/sec"
    << std::endl;

  std::cout << std::fixed;

  std::string labels[6] = {"ompdot", "simdot", "ompmax", "simmax", "ompmin", "simmin"};
  size_t sizes[6] = {
    2 * sizeof(T) * ARRAY_SIZE,
    2 * sizeof(T) * ARRAY_SIZE,
    1 * sizeof(T) * ARRAY_SIZE,
    1 * sizeof(T) * ARRAY_SIZE,
    1 * sizeof(T) * ARRAY_SIZE,
    1 * sizeof(T) * ARRAY_SIZE
  };

  for (int i = 0; i < 6; i++) {
    // Get min/max; ignore the first result
    auto minmax = std::minmax_element(timings[i].begin()+ignore_num_times, timings[i].end());
    int tcount = 0;
    int tmaxiter = 0;
    double tmax = 0.0;
    for (auto tim : timings[i]) {
      if ((tcount>=ignore_num_times) && (tim >= tmax)) { 
        tmax = tim; 
	tmaxiter = tcount;
      }
      tcount++;
    }
    // Display which iteration took the most time
    //printf(" tmax :%f  tmaxiter:%d\n", tmax, tmaxiter);

    // Calculate average; ignore ignore_num_times
    double average = std::accumulate(timings[i].begin()+ignore_num_times, timings[i].end(), 0.0) / (double)(num_times - ignore_num_times);

     printf("  %s       %8.0f   %8.6f  %8.6f   %8.6f    %8.0f\n", labels[i].c_str(),
      1.0E-6 * sizes[i] / (*minmax.first), (double) *minmax.first, (double) *minmax.second,
      (double) average, 1.0E-6 * sizes[i] / (average));
  }
  #pragma omp target exit data map(release: a[0:array_size], b[0:array_size], c[0:array_size])
  free(a);
  free(b);
  free(c);
}
