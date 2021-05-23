#include <complex>
#include <cmath>
#include <iostream>
#include <omp.h>
using namespace std;
#define N 100
int main(int argc, char **argv)
{
  std::complex<float> dinp(0.0,0.0);
  std::complex<float> dres[N];
  for (int i=0; i<N; i++) {
    std::complex<float> dinit(i,1.0);
    dres[i] = dinit;
  }
  #pragma omp target teams distribute parallel for map(to: dres) map(tofrom:dinp) reduction(+:dinp)
  for (int i=0; i<N; i++) 
  {
    dinp += dres[i];
  }
  cout << dinp << '\n';
  if (((float)(N*(N-1))/2.0) != std::real(dinp)) {
    cout << "Failed \n";
    return 1;
  }
  return 0;
}

