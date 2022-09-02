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

#include "ReductionsTestClass.h"

// Default size of 2^25
const int ARRAY_SIZE = 33554432;
//const int ARRAY_SIZE = 33;
unsigned int num_times = 12;
unsigned int ignore_num_times = 2;
unsigned int smoke_rc = 0;

template <typename T, bool >
void run_kernels(const int ARRAY_SIZE);

template <typename T, typename TV>
void run_complex_kernels(const int ARRAY_SIZE);

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

  // Problem with complex math but simulated matches actual reduction
#if 0
  std::cout << std::endl << "TEST DOUBLE COMPLEX" << std::endl;
  run_complex_kernels<double _Complex, double>(ARRAY_SIZE);
  std::cout << std::endl << "TEST FLOAT COMPLEX" << std::endl;
  run_complex_kernels<float _Complex, float>(ARRAY_SIZE);
#endif
  return smoke_rc;
}

template <typename T, bool DO_INT_KERNELS>
void run_kernels(const int ARRAY_SIZE) {
  std::cout << "Running kernels " << num_times << " times" << std::endl;
  std::cout << "Ignoring timing of first "<< ignore_num_times << "  runs " << std::endl;

  double ETOL = 0.0000001;
  T startA, startB, startC;
  startA = 2;
  startB = 3;
  startC = 1;
  if (DO_INT_KERNELS) { 
    std::cout << "Integer Size: " << sizeof(T)  << std::endl;
  } else {
    if (sizeof(T) == sizeof(float))
      std::cout << "Precision: float" << std::endl;
    else
      std::cout << "Precision: double" << std::endl;
  }

  // Create host vectors
  std::vector<T> a(ARRAY_SIZE, startA);
  std::vector<T> b(ARRAY_SIZE, startB);
  std::vector<T> c(ARRAY_SIZE, startC);

  std::cout << "Array elements: " << ARRAY_SIZE << std::endl ;
  std::cout << "Array size:     " << ((ARRAY_SIZE*sizeof(T)) / 1024*1024) << " MB" << std::endl;

  Tc<T> * RTC;
  RTC = new ReductionsTestClass<T>(ARRAY_SIZE);
  RTC->init_arrays(startA, startB, startC);
  T goldSum = (T) (startA*startB) * (T) ARRAY_SIZE;
  T goldMax = (T) startC * (T) ARRAY_SIZE; 
  T goldMin = (T) startC ;
  double goldSum_d = (double) goldSum;
  double goldMax_d = (double) goldMax;
  double goldMin_d = (double) goldMin;

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
      double omp_sum_d = (double) omp_sum;
      double ompErrSum = abs((omp_sum_d - goldSum_d)/goldSum_d);
      if (ompErrSum > ETOL ) {
         std::cerr
        << "omp_dot Validation failed on sum. Error " << ompErrSum << " tol:" << ETOL << std::endl << std::setprecision(15)
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
      double sim_sum_d = (double) sim_sum;
      double simErrSum = abs((sim_sum_d - goldSum_d)/goldSum_d);
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

#include <complex>
#include "ReductionsTestClass_complex.h"
template <typename T, typename TV>
void run_complex_kernels(const int ARRAY_SIZE) {
  std::cout << "Running kernels " << num_times << " times" << std::endl;
  std::cout << "Ignoring timing of first "<< ignore_num_times << "  runs " << std::endl;

  double ETOL = 0.0000001;
  // Use complex conjugate to get 0 imaginary part 
  T startA; __real__(startA) = 1.0; __imag__(startA) = 1.0;
  T startB; __real__(startB) = 1.0; __imag__(startB) = -1.0;
  T startC; __real__(startC) = .4; __imag__(startC) = -4.0;
  if (sizeof(T) == sizeof(float _Complex)) {
    std::cout << "Complex Precision: float size "<< sizeof(float _Complex) << std::endl;
  } else {
    std::cout << "Complex Precision: double size "<< sizeof(double _Complex) << std::endl;
  }

  // Create host vectors
  std::vector<T> a(ARRAY_SIZE, startA);
  std::vector<T> b(ARRAY_SIZE, startB);
  std::vector<T> c(ARRAY_SIZE, startC);

  std::cout << "Array elements: " << ARRAY_SIZE << std::endl ;
  std::cout << "Array size:     " << ((ARRAY_SIZE*sizeof(T)) / 1024*1024) << " MB" << std::endl;

  Tc_complex<T> * RTC;
  RTC = new ReductionsTestClass_complex<T>(ARRAY_SIZE);
  RTC->init_arrays(startA, startB, startC);
  RTC->read_arrays(a, b, c);
  T goldSum;
  __real__(goldSum) = 0.0;
  __imag__(goldSum) = 0.0;
  for (int i = 0; i < ARRAY_SIZE ; i++)
    goldSum += (a[i] * b[i]);

  TV goldSum_r =  __real__(goldSum);
  TV goldSum_i =  __imag__(goldSum);
  printf("Complex sum of products is (%f + %f i)\n",goldSum_r, goldSum_i);

  // List of times
  std::vector<std::vector<double>> timings(2);

  // Declare timers
  std::chrono::high_resolution_clock::time_point t1, t2;

  // Main loop
  for (unsigned int k = 0; k < num_times; k++) {
    t1 = std::chrono::high_resolution_clock::now();
    T omp_sum = RTC->omp_dot();
#if 0
    T omp_sum = 0;
    RTC->read_arrays(a, b, c);
    for (int i = 0; i < ARRAY_SIZE ; i++) {
      T prod;
      TV ar = __real__(a[i]);
      TV ai = __imag__(a[i]);
      TV br = __real__(b[i]);
      TV bi = __imag__(b[i]);
      TV arbr=ar*br;
      TV arbi=ar*bi;
      TV aibr=ai*br;
      TV aibi=ai*bi;
      TV abr = -aibi + arbr; 
      TV abi = arbi + aibr; 
      __real__(prod) = abr; 
      __imag__(prod) = abi;
       omp_sum += prod;
    }
#endif
    t2 = std::chrono::high_resolution_clock::now();
    timings[0].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

#if 0
    TV omp_sum_r = __real__(omp_sum);
    TV omp_sum_i = __imag__(omp_sum);
    TV ompErrSum_r = abs((omp_sum_r - goldSum_r)/goldSum_r);
    TV ompErrSum_i = abs((omp_sum_i - goldSum_i)/goldSum_i);
    // if ((ompErrSum_r > ETOL ) || (ompErrSum_i > ETOL)) {
    if (ompErrSum_r > ETOL ) {  // Imainary sum is 0
      std::cerr 
	<< " Iteration " << k << std::endl 
        << "omp_dot Validation failed on sum. real Error " << ompErrSum_r << " tol:" << ETOL << std::endl 
        << "omp_dot Validation failed on sum. Imag Error " << ompErrSum_i << " tol:" << ETOL << std::endl 
        // << std::endl << std::setprecision(15)
        << "Real Sum was " << omp_sum_r << " but should be " << goldSum_r << std::endl 
        << "Imag Sum was " << omp_sum_i << " but should be " << goldSum_i << std::endl;
      smoke_rc = 1;
    }
#endif
    t1 = std::chrono::high_resolution_clock::now();
    T sim_sum = RTC->sim_dot();
    t2 = std::chrono::high_resolution_clock::now();
    timings[1].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
    TV sim_sum_r = __real__(sim_sum);
    TV sim_sum_i = __imag__(sim_sum);
    TV simErrSum_r = abs((sim_sum_r - goldSum_r)/goldSum_r);
    TV simErrSum_i = abs((sim_sum_i - goldSum_i)/goldSum_i);
    if ((simErrSum_r > ETOL ) || (simErrSum_i > ETOL)) {
    //if (simErrSum_r > ETOL) {
      std::cerr
	<< " Iteration " << k << std::endl 
        << "sim_dot Validation failed on sum. real Error " << simErrSum_r << " tol:" << ETOL << std::endl 
        << "sim_dot Validation failed on sum. Imag Error " << simErrSum_i << " tol:" << ETOL << std::endl 
        // << std::endl << std::setprecision(15)
        << "Real Sum was " << sim_sum_r << " but should be " << goldSum_r << std::endl 
        << "Imag Sum was " << sim_sum_i << " but should be " << goldSum_i << std::endl;
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

  std::string labels[2] = {"ompdot", "simdot"};
  size_t sizes[2] = {
    2 * sizeof(T) * ARRAY_SIZE,
    2 * sizeof(T) * ARRAY_SIZE,
  };

  for (int i = 0; i < 2; i++) {
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

