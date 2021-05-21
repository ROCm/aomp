#include <stdio.h>
#include <stdbool.h>
#include <omp.h>

#define N 1000
#define M 123456

struct TT
{
  int *a;
};

#pragma omp declare target
int a_gbl = 0;
#pragma omp end declare target

int main() {
  bool correct = false;
  int n = N;
  int m = M;
  struct TT p;
  int a = 3;
  int *b = (int *) malloc(n*sizeof(int));
  p.a = (int *) malloc(m*sizeof(int));

  
  #pragma omp target data map(a, b[:n], p.a[:m])
  {
    correct = omp_target_is_present(&a, 0);
    correct = correct && omp_target_is_present(b, 0);
    correct = correct && omp_target_is_present(&p, 0);
    correct = correct && omp_target_is_present(p.a, 0);
  }

  #pragma omp target enter data map(to: a, b[:n], p.a[:m])
  correct = omp_target_is_present(&a, 0);
  correct = correct && omp_target_is_present(b, 0);
  correct = correct && omp_target_is_present(&p, 0);
  correct = correct && omp_target_is_present(p.a, 0);
  #pragma omp target exit data map(from: a, b, p.a)

  correct = correct && omp_target_is_present(&a_gbl, 0);
  
  if (correct)
    printf("All pointers are present on the device. This is correct\n");
  else
    printf("Some pointer is not present on the device. This is wrong\n");

  return correct;
}
