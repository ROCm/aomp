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

int main() {
  const size_t n = 1000;
  const size_t size = 1024*10;
  std::vector<T> vecTs;

  size_t num_devices = omp_get_num_devices();

  for(size_t i = 0; i < n; i++)
    vecTs.emplace_back(T(size));

  // initialize
  #pragma omp target teams distribute parallel for
  for(size_t i = 0; i < n; i++)
    for(size_t j = 0; j < vecTs[i].getSize(); j++)
      vecTs[i].getArr()[j] = (double) i+j;

  // One OpenMP host thread per available GPU, the i-th OpenMP thread offloads to the i-th GPU.
  #pragma omp parallel for num_threads(num_devices)
  for(size_t i = 0; i < vecTs.size(); i++) {
    // no need to map memory in unified_shared_memory.
    #pragma omp target teams distribute parallel for device(omp_get_thread_num())
    for(size_t j = 0; j < vecTs[i].getSize(); j++) {
      vecTs[i].getArr()[j] += (double)j;
    }
  }

  int err = 0;
  for(size_t i = 0; i < vecTs.size(); i++)
    for(size_t j = 0; j < vecTs[i].getSize(); j++) {
      if (vecTs[i].getArr()[j] != (i+2*j)) {
        printf("Error at (%zu,%zu): got %lf, expected %lf\n", i, j, vecTs[i].getArr()[j], (double)(i+2*j));
        err++;
        if (err > 10) return err;
      }
    }

  if (!err) printf("Success!\n");
  return err;
}
