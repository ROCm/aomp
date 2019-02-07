#include <omp.h>
#include <stdio.h>

#define N 1024


int A[N];
int B[N];
int C[N];


int main() 
{
  int i, errors;

  for(i=0; i<N; i++) {
    A[i] = i;
    B[i] = i;
  }
    
  // map data A & B and move to
  #pragma omp target enter data map(to: A, B) depend(out: A[0]) nowait
  
  // no data move since already mapped
  #pragma omp target map(A, B) depend(out: A[0]) nowait
  {
    int i;
    for(int i=0; i<N; i++) A[i]++;
    for(int i=0; i<N; i++) B[i]++;
  }
  
  // no data move since already mapped
  #pragma omp target map(A, B) depend(out: A[0]) nowait
  {
    int i;
    for(int i=0; i<N; i++) A[i]++;
    for(int i=0; i<N; i++) B[i]++;
  }

  // A updated via update
  #pragma omp target update from(A) depend(out: A[0]) nowait

  // B updated via exit, A just released
  #pragma omp target exit data map(release: A) map(from: B) depend(out: A[0]) nowait

  
  #pragma omp taskwait


  errors = 0;
  for(i=0; i<N; i++) {
    if (A[i] != i+2) printf("%d: A got %d, expected %d; error %d\n", i, A[i], i+2, ++errors);
    if (B[i] != i+2) printf("%d: B got %d, expected %d; error %d\n", i, B[i], i+2, ++errors);
    if (errors>25) break;
  }
  printf("completed with %d errors\n", errors);
  return 1;   
}
