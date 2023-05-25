// Using rocm 5.5 RC1 (aka rocm5.5.0beta1)
// Build with:
// $ amdclang++ \
//      -fopenmp \
//      -fopenmp-targets=amdgcn-amd-amdhsa \
//      -Xopenmp-target=amdgcn-amd-amdhsa \
//      -march=gfx90a \
//      ELCAP-321.cc
//
//
// Fails at runtime with the following message:
//
// Libomptarget error: Unable to generate entries table for device id 0.
// Libomptarget error: Failed to init globals on device 0
// Libomptarget error: Consult https://openmp.llvm.org/design/Runtimes.html for debugging options.
// Libomptarget error: Source location information not present. Compile with -g or -gline-tables-only.
// Libomptarget fatal error 1: failure of target construct while offloading is mandatory
// Abort (core dumped)



#include <cstdio>
#include <iostream>
#include <omp.h>

#pragma omp declare target
class IMCConstants
{
public:
  constexpr static double pi = 3.141592653589793238462; // 4.*atan(1.0);         
  IMCConstants() {;}
  ~IMCConstants() {;}
};
#pragma omp end declare target

template<typename T>
class Array1D {
public:
  Array1D(int _size, T* _data) : 
      size(_size), 
      data(_data) 
  {
     // set storage
#pragma omp target enter data map(to:this[:1])
#pragma omp target enter data map(to:data[0:size])
  }

    ~Array1D() {
#pragma omp target exit data map(release:this[0].data[:size])
#pragma omp target exit data map(release:this[:1])
    }

    T& operator[](int i) {
      return data[i];
    }

    int get_size() const {
      return size;;
    }
    double get_pi() const {
      return IMCConstants::pi;
    }
  

private:
  int size;
  T* data; // host array
};

#define N 10

int main(int argc, char **argv) {
  int * data = new int[N];
    for (int i = 0; i < N; i++) {
      data[i] = i;
    }
  double * ddata = new double[N];
    for (int i = 0; i < N; i++) {
      ddata[i] = i;
    }
    Array1D<int> a1D(N, data);
    Array1D<double> a1Dd(N, ddata);

    // use target region to modify array on device
    #pragma omp target 
    {
    for (int i = 0; i < a1D.get_size(); i++) {
        a1D[i] *= 2;
        a1Dd[i] *= 2.*a1Dd.get_pi();
    }
    }
    for (int i = 0; i < N; i++) {
      printf("HOST  a1D[%i] = %i\n", i,a1D[i]);
    }
    for (int i = 0; i < N; i++) {
      printf("HOST  a1Dd[%i] = %e\n", i,a1Dd[i]);
    }
    // print modified array
    #pragma omp target 
    {
    for (int i = 0; i < N; i++) {
      printf("mod DEVICE  a1D[%i] = %i\n", i,a1D[i]);
    }
    for (int i = 0; i < N; i++) {
      printf("mod DEVICE  a1Dd[%i] = %e\n", i,a1Dd[i]);
    }
    }
   std::cout << std::endl;

    delete [] data;
    delete [] ddata;

    return 0;
}
