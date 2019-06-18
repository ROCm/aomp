#include <cmath>
 
int main(int argc, char ** argv) {
  double x0 = 1.0;
  double sumx = 0.0;
 
  #pragma omp target enter data map(to:x0)
 
  #pragma omp target teams distribute parallel for map(tofrom: sumx) reduction(+:sumx)
  for (std::size_t i = 0; i < 1000; ++i ) {
    sumx += x0;
  }
 
  #pragma omp target exit data map(delete: x0)
}
 

