#include<cstdio>
#include<vector>

#include<omp.h>

#pragma omp requires unified_shared_memory

class T {
public:
  T(int size) : size(size) {
    arr = new double[size];
  }
  T(T&&t) {
    arr = t.arr;
    size = t.size;
  }
  ~T() {}

  size_t getSize() { return size; };
  double * getArr() { return arr; }
private:
  double *arr;
  size_t size;
};

#define N 1000
#define ARR_SIZE 100

int main() {
  size_t n = N;
  size_t size = ARR_SIZE;
  std::vector<T> vecTs;

  for(size_t i = 0; i < n; i++)
    vecTs.emplace_back(T(size));

  // initialize
  #pragma omp target teams distribute parallel for
  for(size_t i = 0; i < n; i++)
    for(size_t j = 0; j < vecTs[i].getSize(); j++)
      vecTs[i].getArr()[j] = (double) i+j;

  // compute with multiple GPU kernels, one per available device
  #pragma omp parallel for num_threads(omp_get_num_devices())
  for(int i = 0; i < vecTs.size(); i++) {
    printf("TID = %d/%d: offloading to a GPU\n", omp_get_thread_num(), omp_get_num_threads());
    #pragma omp target teams distribute parallel for device(omp_get_thread_num()) // map(tofrom: vecTs[i].getArr[:vecTs[i].getSize()])
    for(int j = 0; j < vecTs[i].getSize(); j++) {
      if (i == 0 && j == 10)
	vecTs[i].getArr()[j] = (double)3.14;
      else
	vecTs[i].getArr()[j] += (double)j;
    }
  }
  printf("%lf\n", vecTs[0].getArr()[10]);
  return 0;
}
