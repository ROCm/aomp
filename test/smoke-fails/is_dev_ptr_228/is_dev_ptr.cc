#include "hip/hip_runtime.h"
#include <iostream>

class Foo {
public:
  Foo(int size);
  void init();
  bool verify();

private:
  int _size;
  int *_data;
};

Foo::Foo(int size) : _size(size) {
  // ignore err
  hipError_t err = hipMalloc(&_data, _size * sizeof(int));
}

void Foo::init() {
  printf("_data = %p\n", _data);
#pragma omp target teams distribute parallel for is_device_ptr(_data)
  // #pragma omp target teams distribute parallel for
  for (int ii = 0; ii < _size; ++ii) {
    if (ii == 0)
      printf("_data = %p\n", _data);
    _data[ii] = ii;
  }
}

bool Foo::verify() {
  for (int ii = 0; ii < _size; ++ii)
    if (_data[ii] != ii)
      return false;
  return true;
}

int main(int argc, char **argv) {
  Foo foo(100);
  foo.init();
  if (foo.verify())
    std::cout << "PASS" << std::endl;
  else
    std::cout << "FAIL" << std::endl;
}
