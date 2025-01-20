#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define NUM_TEAMS 250
#define NUM_THREADS 256
#define N NUM_THREADS *NUM_TEAMS

template<typename T>
void run_test() {
  T *in = (T*)malloc(sizeof(T) * N);
  T *out1 = (T*)malloc(sizeof(T) * N);  // For inclusive scan
  T *out2 = (T *)malloc(sizeof(T) * N); // For exclusive scan

  for (int i = 0; i < N; i++) {
    in[i] = 10;
    out1[i] = 0;
  }

  T sum1 = T(0);

#pragma omp target teams distribute parallel for reduction(inscan, +:sum1) map(tofrom: in[0:N], out1[0:N]) num_teams(NUM_TEAMS) num_threads(NUM_THREADS)
  for (int i = 0; i < N; i++) {
    sum1 += in[i]; // input phase
#pragma omp scan inclusive(sum1)
    out1[i] = sum1; // scan phase
  }

  T checksum = T(0);
  for (int i = 0; i < N; i++) {
    checksum += in[i];
    if (checksum != out1[i]) {
      printf("Inclusive Scan: Failure. Wrong Result at %d. Exiting...\n", i);
      return;
    }
  }
  free(out1);
  printf("Inclusive Scan: Success!\n");

  T sum2 = T(0);

#pragma omp target teams distribute parallel for reduction(inscan, +:sum2) map(tofrom: in[0:N], out2[0:N]) num_teams(NUM_TEAMS) num_threads(NUM_THREADS)
  for (int i = 0; i < N; i++) {
    out2[i] = sum2; // scan phase
#pragma omp scan exclusive(sum2)
    sum2 += in[i]; // input phase
  }

  checksum = T(0);
  for (int i = 0; i < N; i++) {
    if (checksum != out2[i]) {
      printf("Exclusive Scan: Failure. Wrong Result at %d. Exiting...\n", i);
      return;
    }
    checksum += in[i];
  }
  free(in);
  free(out2);
  printf("Exclusive Scan: Success!\n");
}

int main() {
  printf("Testing for datatype int\n");
  run_test<int>();
  
  printf("Testing for datatype uint32_t\n");
  run_test<uint32_t>();

  printf("Testing for datatype uint64_t\n");
  run_test<uint64_t>();

  printf("Testing for datatype long\n");
  run_test<long>();

  printf("Testing for datatype double\n");
  run_test<double>();
  
  printf("Testing for datatype float\n");
  run_test<float>();
  return 0;
}

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args: 9 teamsXthrds:( 250X 256)
/// CHECK: n:__omp_offloading_[[MANGLED:.*i.*]]_l22

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args: 9 teamsXthrds:( 250X 256)
/// CHECK: n:__omp_offloading_[[MANGLED]]_l22_1

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args: 9 teamsXthrds:( 250X 256)
/// CHECK: n:__omp_offloading_[[MANGLED:.*i.*]]_l42

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args: 9 teamsXthrds:( 250X 256)
/// CHECK: n:__omp_offloading_[[MANGLED]]_l42_1

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args: 9 teamsXthrds:( 250X 256)
/// CHECK: n:__omp_offloading_[[MANGLED:.*j.*]]_l22

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args: 9 teamsXthrds:( 250X 256)
/// CHECK: n:__omp_offloading_[[MANGLED]]_l22_1

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args: 9 teamsXthrds:( 250X 256)
/// CHECK: n:__omp_offloading_[[MANGLED:.*j.*]]_l42

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args: 9 teamsXthrds:( 250X 256)
/// CHECK: n:__omp_offloading_[[MANGLED]]_l42_1

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args: 9 teamsXthrds:( 250X 256)
/// CHECK: n:__omp_offloading_[[MANGLED:.*m.*]]_l22

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args: 9 teamsXthrds:( 250X 256)
/// CHECK: n:__omp_offloading_[[MANGLED]]_l22_1

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args: 9 teamsXthrds:( 250X 256)
/// CHECK: n:__omp_offloading_[[MANGLED:.*m.*]]_l42

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args: 9 teamsXthrds:( 250X 256)
/// CHECK: n:__omp_offloading_[[MANGLED]]_l42_1

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args: 9 teamsXthrds:( 250X 256)
/// CHECK: n:__omp_offloading_[[MANGLED:.*l.*]]_l22

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args: 9 teamsXthrds:( 250X 256)
/// CHECK: n:__omp_offloading_[[MANGLED]]_l22_1

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args: 9 teamsXthrds:( 250X 256)
/// CHECK: n:__omp_offloading_[[MANGLED:.*l.*]]_l42

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args: 9 teamsXthrds:( 250X 256)
/// CHECK: n:__omp_offloading_[[MANGLED]]_l42_1

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args: 9 teamsXthrds:( 250X 256)
/// CHECK: n:__omp_offloading_[[MANGLED:.*d.*]]_l22

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args: 9 teamsXthrds:( 250X 256)
/// CHECK: n:__omp_offloading_[[MANGLED]]_l22_1

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args: 9 teamsXthrds:( 250X 256)
/// CHECK: n:__omp_offloading_[[MANGLED:.*d.*]]_l42

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args: 9 teamsXthrds:( 250X 256)
/// CHECK: n:__omp_offloading_[[MANGLED]]_l42_1

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args: 9 teamsXthrds:( 250X 256)
/// CHECK: n:__omp_offloading_[[MANGLED:.*f.*]]_l22

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args: 9 teamsXthrds:( 250X 256)
/// CHECK: n:__omp_offloading_[[MANGLED]]_l22_1

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args: 9 teamsXthrds:( 250X 256)
/// CHECK: n:__omp_offloading_[[MANGLED:.*f.*]]_l42

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: args: 9 teamsXthrds:( 250X 256)
/// CHECK: n:__omp_offloading_[[MANGLED]]_l42_1

/// CHECK: Testing for datatype int
/// CHECK: Inclusive Scan: Success!
/// CHECK: Exclusive Scan: Success!

/// CHECK: Testing for datatype uint32_t
/// CHECK: Inclusive Scan: Success!
/// CHECK: Exclusive Scan: Success!

/// CHECK: Testing for datatype uint64_t
/// CHECK: Inclusive Scan: Success!
/// CHECK: Exclusive Scan: Success!

/// CHECK: Testing for datatype long
/// CHECK: Inclusive Scan: Success!
/// CHECK: Exclusive Scan: Success!

/// CHECK: Testing for datatype double
/// CHECK: Inclusive Scan: Success!
/// CHECK: Exclusive Scan: Success!

/// CHECK: Testing for datatype float
/// CHECK: Inclusive Scan: Success!
/// CHECK: Exclusive Scan: Success!