#include<cstdio>

#pragma omp begin declare target
#pragma omp begin assumes no_openmp
void foo(int *a, int i) {
  a[i] = i;
}
#pragma opm end assumes
#pragma omp end declare target

#define N 100000

int main() {
  int *a = new int[N];

  #pragma omp target teams distribute parallel for map(from: a[:N])
  for(int i = 0; i < N; i++) {
    foo(a,i);
  }

  int err = 0;
  for(int i = 0; i < N; i++)
    if (a[i] != i) {
      err++;
      printf("Error at %d: got %d expected %d\n", i, a[i], i);
      if (err > 10) return err;
    }
  return err;

}
