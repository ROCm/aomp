#include "cstdlib"
#include <stdio.h>
#include <omp.h>
void copyArr(int *c)
{
 int n = 10000;
 #pragma omp target map(from:c)
 {
   c=(int *)malloc(n*sizeof(int));
 }
}
int copy_test() {
  int *c = nullptr;
  copyArr(c);
  printf("Pass\n");
  return 0;
}
class foo{
  public:
  int num;
  foo(int n) {        
    num = n;
  }
};
int main()
{
  const int n=32;
  int *c=NULL;
  foo *f;
  #pragma omp target map(from:c) 
  {
    c=(int *)malloc(n*sizeof(int));
    free(c);
    f = new foo(5);
    printf("foo: %d\n", f->num);
    delete(f);
  }
  copy_test();
  printf("Passed!\n");
}
 


