#include <cstdlib>
#include <iostream>

extern "C" void consume(int *p);

int identity(int i) { return i; }

/// @brief This version maps one larger array and works on each entry as opposed
/// to individual integers
/// @param argc
/// @param argv
/// @return
int main(int argc, char **argv) {
  const int threadParallel = 32;
  const int deviceComputeLoad = 16;

// For threadParallel == 1 && deviceComputeLoad == 1 -> 9 Events observed
#pragma omp parallel for schedule(static,1)
  for (int i = 0; i < threadParallel; ++i) {
    int vals[deviceComputeLoad] = {0};
    int vals2[deviceComputeLoad] = {0};
    int vals3[deviceComputeLoad] = {0};

// Generates 2 data-in events, 2 data-out events
#pragma omp target data map(vals,vals2)
{
  // Generates 1 kernel launch event
#pragma omp target teams distribute
    for (int j = 0; j < deviceComputeLoad; ++j) {
      vals[j] += 1;
    }
// Generates 1 kernel launch event
#pragma omp target teams distribute
    for (int j = 0; j < deviceComputeLoad; ++j) {
      vals2[j] += 1;
    }
}

// Generates 1 data-in event, 1 data-out event, 1 kernel launch event
#pragma omp target teams distribute map(vals3)
    for(int j = 0; j< deviceComputeLoad; ++j) {
      vals3[j] += 2;
    }

    for (int j = 0; j < deviceComputeLoad; ++j) {
      if (vals3[j] != 2) {
        std::cout << "MISMATCH " << vals3[j] << std::endl;
        abort();
      }
    }
  }

  std::cout << "Success" << std::endl;

  return 0;
}
