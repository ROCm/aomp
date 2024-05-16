#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <chrono>
#include <omp.h>

//#pragma omp requires unified_shared_memory

using namespace std::chrono;

void runConfig(size_t numDoubles, unsigned int repeats, std::vector<double>&allocTime, std::vector<double>&copyTime, std::vector<double>&freeTime) {
  double *hostMem = nullptr, *devMem = nullptr;
  int devId = omp_get_default_device();
  for (int i = 0; i < repeats; i++) {
    {
      high_resolution_clock::time_point startAlloc = high_resolution_clock::now();
      
      hostMem = (double *)malloc(numDoubles*sizeof(double));

      /// This is done in the OpenMP runtime upon copy
      #if 0
      switch (config) {
      case CONFIG_2:
	hipMemAdvise(hostMem, numDoubles*sizeof(double), hipMemAdviseSetAccessedBy, devId);
	break;
      case CONFIG_3:
	hipMemPrefetchAsync(hostMem, numDoubles*sizeof(double), devId);
	hipDeviceSynchronize();
	break;
      default:
	break;
      }
      #endif
      
      devMem = (double *)omp_target_alloc(numDoubles*sizeof(double), devId);

      high_resolution_clock::time_point endAlloc = high_resolution_clock::now();
      duration<double> allocElapsed = duration_cast<duration<double>>(endAlloc - startAlloc);
      allocTime.push_back(allocElapsed.count());
      if (!devMem) {
	continue;
      }
    }

    {
      high_resolution_clock::time_point startCopy = high_resolution_clock::now();

      int err = omp_target_memcpy(devMem, hostMem, numDoubles*sizeof(double), /*dst_offset=*/0, /*src_offset=*/0, omp_get_default_device(), omp_get_initial_device());

      high_resolution_clock::time_point endCopy = high_resolution_clock::now();
      duration<double> copyElapsed = duration_cast<duration<double>>(endCopy - startCopy);
      copyTime.push_back(copyElapsed.count());
      if (err != 0) {
	printf("Error copying\n");
      }
    }

    {
      high_resolution_clock::time_point startFree = high_resolution_clock::now();

      free(hostMem);
      omp_target_free(devMem, omp_get_default_device());

      high_resolution_clock::time_point endFree = high_resolution_clock::now();
      duration<double> freeElapsed = duration_cast<duration<double>>(endFree - startFree);

      freeTime.push_back(freeElapsed.count());
    }
  }
}

void printToCSVFile(size_t numDoubles, unsigned int repeats, std::ofstream &outfile, std::vector<double> allocTime, std::vector<double> copyTime, std::vector<double> freeTime) {
  outfile << std::fixed;
  outfile << std::setprecision(20);

  double sumAlloc = 0.0;
  double sumCopy = 0.0;
  double sumFree = 0.0;
  for (int i = 0; i < repeats; i++) {
    sumAlloc += allocTime[i];
    sumCopy += copyTime[i];
    sumFree += freeTime[i];
  }
  double averageAlloc = sumAlloc/repeats;
  double averageCopy = sumCopy/repeats;
  double averageFree = sumFree/repeats;

  std::string sizeStr = std::to_string(numDoubles*sizeof(double));
  outfile << sizeStr << ", ";
  for (int i = 0; i < repeats; i++)
    outfile << allocTime[i] << ", ";
  outfile << averageAlloc << ", ";
  for (int i = 0; i < repeats; i++)
    outfile << copyTime[i] << ", ";
  outfile << averageCopy << ", ";
  for (int i = 0; i < repeats; i++)
    outfile << freeTime[i] << ", ";
  outfile << averageFree << ", ";
  outfile << "\n";
}

void runConfigs(std::vector<size_t>&numDoublesVect, bool isXnackEnabled, size_t repeats) {
  {
    std::string fileName = "omp.";
    if (isXnackEnabled)
      fileName += "xnack_enabled";
    else
      fileName +="xnack_disabled";
    fileName += ".csv";
    std::ofstream outfile (fileName);

    outfile << "Size, Alloc 0, 1, 2, 3, 4, 5, 6, 7, Avg, Copy 0, 1, 2, 3, 4, 5, 6, 7, Avg, Free 0, 1, 2, 3, 4, 5, 6, 7, Avg\n";

    for(auto numDoubles: numDoublesVect) {
      std::vector<double> allocTime;
      std::vector<double> copyTime;
      std::vector<double> freeTime;
      runConfig(numDoubles, repeats, allocTime, copyTime, freeTime);
      printToCSVFile(numDoubles, repeats, outfile, allocTime, copyTime, freeTime);
    }
  }
}

int main() {
  const size_t magnitudeBase = 1024;
  const size_t multiplierBase = 2;
  std::vector<size_t> numDoublesVect;

  for (size_t magnitudeExponent = 0; magnitudeExponent < 4; magnitudeExponent++) {
    const size_t magnitude = pow(magnitudeBase, magnitudeExponent); // B, KB, MB, GB
    for (size_t multiplierExponent = 0; multiplierExponent < 10; multiplierExponent++) {
      const size_t multiplier = pow(multiplierBase, multiplierExponent); // 1, 2, 4, .., 1024
      if (magnitude == 1 && multiplier < 8) continue;
      if (magnitude == 1024*1024*1024 && multiplierExponent > 5) continue;
      const size_t arraySize = multiplier*magnitude; // 1B, 2B, .., 1KB, 2KB, ...
      const size_t numDoubles = arraySize/8;
      numDoublesVect.push_back(numDoubles);

      //#ifdef PRINT_SIZES
      printf("ArraySize in bytes = %lu = %lu^%lu * %lu^%lu == %lu double's\n", arraySize, multiplierBase, multiplierExponent, magnitudeBase, magnitudeExponent, arraySize/8);
      //#endif
    }
  }

  bool isXnackEnabled = false;
  const char* xnack = std::getenv("HSA_XNACK");
  if (xnack != nullptr) {
    int xnackValue = atoi(xnack);
    if (xnackValue != 0)
      isXnackEnabled = true;
  }
  std::cout << std::fixed;
  std::cout << std::setprecision(12);

  runConfigs(numDoublesVect, isXnackEnabled, /*repeats=*/8);

  return 0;
}
