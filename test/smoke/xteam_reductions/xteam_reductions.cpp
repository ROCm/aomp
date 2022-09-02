#include <omp.h>
#include <cstdlib>  // For aligned_alloc
#include <vector>

#ifndef ALIGNMENT
#define ALIGNMENT (2*1024*1024) // 2MB
#endif

#include <iostream>
#include <stdexcept>

#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <limits>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cstring>

#include "ReductionsTestClass.h"

// Default size of 2^25
const int ARRAY_SIZE = 33554432;
unsigned int num_times = 12;
unsigned int ignore_num_times = 2;
unsigned int smoke_rc = 0;

template <typename T, bool >
void run_kernels(const int ARRAY_SIZE);

int main(int argc, char *argv[]) {
  run_kernels<double,false>(ARRAY_SIZE);
  run_kernels<float,false>(ARRAY_SIZE);
  run_kernels<int,true>(ARRAY_SIZE);
  // We still need to develop sum, max, min for these data types
  //run_kernels<unsigned int,true>(ARRAY_SIZE);
  //run_kernels<long,true>(ARRAY_SIZE);
  //run_kernels<lunsigned ong,true>(ARRAY_SIZE);
  return smoke_rc;
}

template <typename T, bool DO_INT_KERNELS>
void run_kernels(const int ARRAY_SIZE) {
  std::cout << std::endl << "Running kernels " << num_times << " times" << std::endl;
  std::cout << "Ignoring timing of first "<< ignore_num_times << "  runs " << std::endl;

  T ETOL,startA, startB, startC;
  startA = 2;
  startB = 3;
  startC = 1;
  if (DO_INT_KERNELS) { 
    std::cout << "Integer Size: " << sizeof(T)  << std::endl;
  } else {
    ETOL = 0.0000001;
    if (sizeof(T) == sizeof(float))
      std::cout << "Precision: float" << std::endl;
    else
      std::cout << "Precision: double" << std::endl;
  }
  std::vector<T> a(ARRAY_SIZE, startA);
  std::vector<T> b(ARRAY_SIZE, startB);
  std::vector<T> c(ARRAY_SIZE, startC);

  // Create host vectors
  std::streamsize ss = std::cout.precision();
  std::cout << std::setprecision(1) << std::fixed
    << "Array size: " << ARRAY_SIZE*sizeof(T)*1.0E-6 << " MB"
    << " (=" << ARRAY_SIZE*sizeof(T)*1.0E-9 << " GB)" << std::endl;
  std::cout << "Total size: " << 3.0*ARRAY_SIZE*sizeof(T)*1.0E-6 << " MB"
    << " (=" << 3.0*ARRAY_SIZE*sizeof(T)*1.0E-9 << " GB)" << std::endl;
  std::cout.precision(ss);

  Tc<T> * RTC;
  RTC = new ReductionsTestClass<T>(ARRAY_SIZE);
  RTC->init_arrays(startA, startB, startC);
  T goldSum = (T) (startA*startB) * (T) ARRAY_SIZE;
  T goldMax = (T) startC * (T) ARRAY_SIZE; 
  T goldMin = (T) startC ;

  // List of times
  std::vector<std::vector<double>> timings(6);

  // Declare timers
  std::chrono::high_resolution_clock::time_point t1, t2;

  // Main loop
  for (unsigned int k = 0; k < num_times; k++) {
    t1 = std::chrono::high_resolution_clock::now();
    T omp_sum = RTC->omp_dot();
    t2 = std::chrono::high_resolution_clock::now();
    timings[0].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
    if (DO_INT_KERNELS) {
      if (omp_sum != goldSum) {
         std::cerr
        << "omp_dot Validation failed on sum. "
        << "Sum was " << omp_sum << " but should be " << goldSum
        << std::endl;
         smoke_rc = 1;
      }
    } else {
      T ompErrSum = abs((omp_sum - goldSum)/goldSum);
      if (ompErrSum > ETOL ) {
         std::cerr
        << "omp_dot Validation failed on sum. Error " << ompErrSum << " tol:" << ETOL
        << std::endl << std::setprecision(15)
        << "Sum was " << omp_sum << " but should be " << goldSum
        << std::endl;
         smoke_rc = 1;
      }
    }

    t1 = std::chrono::high_resolution_clock::now();
    T sim_sum = RTC->sim_dot();
    t2 = std::chrono::high_resolution_clock::now();
    timings[1].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

    if (DO_INT_KERNELS) {
      if (sim_sum != goldSum) {
         std::cerr
        << "sim_dot Validation failed on sum. "
        << "Sum was " << sim_sum << " but should be " << goldSum
        << std::endl;
         smoke_rc = 1;
      }
    } else {
      T simErrSum = abs((sim_sum - goldSum)/goldSum);
      if (simErrSum > ETOL) {
         std::cerr
        << "sim_dot Validation failed on sum. Error " << simErrSum << " tol:" << ETOL
        << std::endl << std::setprecision(15)
        << "Sum was " << sim_sum << " but should be " << goldSum
        << std::endl;
         smoke_rc = 1;
      } 
    } 

    t1 = std::chrono::high_resolution_clock::now();
    T omp_max = RTC->omp_max();
    t2 = std::chrono::high_resolution_clock::now();
    timings[2].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
    if (omp_max != goldMax) {
      std::cerr
      << "omp_max Validation failed . " << std::endl << std::setprecision(15)
      << "max was " << omp_max << " but should be " << goldMax << std::endl;
      smoke_rc = 1;
    }

    t1 = std::chrono::high_resolution_clock::now();
    T sim_max = RTC->sim_max();
    t2 = std::chrono::high_resolution_clock::now();
    timings[3].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
    if (sim_max != goldMax) {
      std::cerr
      << "sim_max Validation failed . " << std::endl << std::setprecision(15)
      << "max was " << sim_max << " but should be " << goldMax << std::endl;
      smoke_rc = 1;
    }

    t1 = std::chrono::high_resolution_clock::now();
    T omp_min = RTC->omp_min();
    t2 = std::chrono::high_resolution_clock::now();
    timings[4].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
    if (omp_min != goldMin) {
      std::cerr
      << "omp_min Validation failed . " << std::endl << std::setprecision(15)
      << "min was " << omp_min << " but should be " << goldMin << std::endl;
      smoke_rc = 1;
    }


    t1 = std::chrono::high_resolution_clock::now();
    T sim_min = RTC->sim_min();
    t2 = std::chrono::high_resolution_clock::now();
    timings[5].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
    if (sim_min != goldMin) {
      std::cerr
      << "sim_min Validation failed . " << std::endl << std::setprecision(15)
      << "min was " << sim_min << " but should be " << goldMin << std::endl;
      smoke_rc = 1;
    }
     
  }  // end for (unsigned int k = 0; k < num_times; k++)

  RTC->read_arrays(a, b, c);

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
  delete RTC;
}

