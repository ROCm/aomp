

#pragma omp begin declare target
void foo(double *a, int i) {
  a[i] = i;
}
#pragma omp end declare target

#define N 100000

int main() {
  int *a = new int[N];

  #pragma omp target teams distribute parallel for map(from: a[:N])
  for(int i = 0; i < N; i++) {
    #pragma omp assume no_openmp
    {
    foo(a,i);
    }
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
