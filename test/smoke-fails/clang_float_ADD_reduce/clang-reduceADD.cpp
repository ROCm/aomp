#include <iostream>
#include <chrono>
#include <limits>

#include "utilities.hpp"

using namespace std;

#define DTYPE float

int main(int argc, char** argv)
{
  using util = ADD<DTYPE>;

  long elements = 2 *1024 * 1024 / sizeof(DTYPE);
  DTYPE *data = (DTYPE*) malloc(sizeof(DTYPE)*elements);
  DTYPE out = util::init();

#pragma omp target data map (alloc:data[0:elements])
  {
#pragma omp target teams distribute parallel for
    for ( long i = 0 ; i < elements; i++){
      data[i] = util::init(i, elements);// + util::init();
    }

    auto start = chrono::high_resolution_clock::now();
#pragma omp target teams distribute parallel for reduction(+:out) 
    for ( long i = 0 ; i < elements; i++){
      out = util::OP(out, data[i]);
    }
    auto end = chrono::high_resolution_clock::now();
    cout << "ELAPSED TIME: "
      << chrono::duration<double>(end - start).count() <<  endl;
  }

  std::cout<< "SIZE OF ELEMENT:" << sizeof(DTYPE) << "\n"; 
  std::cout<< "REDUCTION TYPE:" << util::info() << "\n";

  if ( !util::validate(out, elements) ){
    std::cout << "Value is " << out << "\n";
    std::cout << "FAIL\n";
    return -1;
  }

  std::cout << "PASS \n";
  free(data);

  return 0;
}

