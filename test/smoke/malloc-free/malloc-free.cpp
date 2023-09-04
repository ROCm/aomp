#include <cstdio>

struct S {
  S(int x) : x(x) {}
  int x;
};

int main() {
  #pragma omp target
  {
    S *ps = new S(3);
    delete ps;
  }

  S *p = nullptr;
  #pragma omp target map(from:p)
  {
    S *ps = new S(2);
    p = ps;
  }
  if (p == nullptr) {
    printf("p is nullptr on host\n");
    return 1;
  }

  #pragma omp target is_device_ptr(p)
  {
    delete p;
  }
  return 0;
}
