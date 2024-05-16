#include <cstdio>

class A {
public:
  A(int konst) : konst(konst) {}
  virtual int foo(int x) = 0;
protected:
  int konst;
};

class B : public A {
public:
  B(int konst) : A(konst) {}
  virtual int foo(int x) {
    return x-konst;
  }
};

class C : public B {
public:
  C(int konst) : B(konst) {}
  virtual int foo(int x) {
    return x+konst;
  }
};

#pragma omp declare target
C gbl_C(10);
#pragma omp end declare target

int main() {
  C host_C(200);

  int g, h, d;
  g = h = d = -1;
  
  #pragma omp target map(from:g, h, d)
  {
    C dev_C(3000);

    g = gbl_C.foo(3);
    h = host_C.foo(3);
    d = dev_C.foo(3);
    printf("global virtual function gbl_C.foo(3) = (should be %d): %d\n", 13, g);
    printf("host virtual function host_C.foo(3) = (should be %d): %d\n", 203, h);
    printf("device virtual function dev_C.foo(3) = (should be %d): %d\n", 3003, d);
  }

  int err = 0;
  if (g != 13 || h != 203 || d != 3003) {
    printf("Error!\n");
    err = 1;
  }
  
  return err;
}

