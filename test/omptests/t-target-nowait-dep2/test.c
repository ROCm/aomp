#include <omp.h>
#include <stdio.h>

#define N 1024

#define EXPLICIT_TARGET_TASK 0

int A[N];
int B[N];
int C[N];
int D[N];
int Z[N+1];

#if EXPLICIT_TARGET_TASK
  #define LOMP_TASK_DEP_40 1
  #define LOMP_TARGET_40 1
  #define LOMP_PROC_BIND_40 1 
  #define LOMP_OS_LINUX 1
  #define LOMP_CANCEL_40 1
  #include "/gsa/yktgsa-h1/00/eichen/new-tlomp/lomp/include/omp_interface.h"
 
  void _of1(lomp_Handle handle, char *fparg, char *sharg)
  {
    #pragma omp target nowait map(A)
    {
      int i;
      for(i=0; i<N; i++) A[i]++;
    }
  }

  void _of2(lomp_Handle handle, char *fparg, char *sharg)
  {
    #pragma omp target nowait map(A, B)
    {
      int i;
      for(i=0; i<N; i++) B[i] += A[i];
    }
  }

  void _of3(lomp_Handle handle, char *fparg, char *sharg)
  {
    #pragma omp target nowait map(A, C)
    {
      int i;
      for(i=0; i<N; i++) C[i] += A[i];
    }
  }

  void _of4(lomp_Handle handle, char *fparg, char *sharg)
  {
    #pragma omp target nowait map(A, B, C)
    {
      int i;
      for(i=0; i<N; i++) A[i] = B[i] + C[i];
    }
  }
#endif

int main() 
{
  int i, errors;

  for(i=0; i<N; i++) {
    A[i] = i;
    B[i] = i+1;
    C[i] = i+2;
    D[i] = i+3;
    Z[i] = 1;
  }
  // set this to zero for fast computation, large number for bigger number
  Z[N] = 1024;

#pragma omp target data map(A, B, C, D, Z) 
  {    

    #if EXPLICIT_TARGET_TASK
      lomp_Handle h = _lomp_GetHandle();

      // 1
      lomp_TaskDep_Public *taskDepArray1;
      void *fpagr1 = _lomp_Task_AllocateFirstPrivate_WithDeps(h, 0, 1, &taskDepArray1);
      // set dependences
      taskDepArray1[0].addr   = &A[0];
      taskDepArray1[0].status = LOMP_TASK_DEP_STATUS_OUT;
      // launch target task
      _lomp_TargetTask_Setup_WithDep(0, h, _of1, fpagr1, NULL, 0, 1, taskDepArray1, 0);

      // 2
      lomp_TaskDep_Public *taskDepArray2;
      void *fpagr2 = _lomp_Task_AllocateFirstPrivate_WithDeps(h, 0, 2, &taskDepArray2);
      // set dependences
      taskDepArray2[0].addr   = &A[0];
      taskDepArray2[0].status = LOMP_TASK_DEP_STATUS_IN;
      taskDepArray2[1].addr   = &B[0];
      taskDepArray2[1].status = LOMP_TASK_DEP_STATUS_OUT;
      // launch target task
      _lomp_TargetTask_Setup_WithDep(0, h, _of2, fpagr2, NULL, 0, 2, taskDepArray2, 0);

      // 3
      lomp_TaskDep_Public *taskDepArray3;
      void *fpagr3 = _lomp_Task_AllocateFirstPrivate_WithDeps(h, 0, 2, &taskDepArray3);
      // set dependences
      taskDepArray3[0].addr   = &A[0];
      taskDepArray3[0].status = LOMP_TASK_DEP_STATUS_IN;
      taskDepArray3[1].addr   = &C[0];
      taskDepArray3[1].status = LOMP_TASK_DEP_STATUS_OUT;
      // launch target task
      _lomp_TargetTask_Setup_WithDep(0, h, _of3, fpagr3, NULL, 0, 2, taskDepArray3, 0);

      // 4
      lomp_TaskDep_Public *taskDepArray4;
      void *fpagr4 = _lomp_Task_AllocateFirstPrivate_WithDeps(h, 0, 2, &taskDepArray4);
      // set dependences
      taskDepArray4[0].addr   = &B[0];
      taskDepArray4[0].status = LOMP_TASK_DEP_STATUS_OUT;
      taskDepArray4[1].addr   = &C[0];
      taskDepArray4[1].status = LOMP_TASK_DEP_STATUS_OUT;
      // launch target task
      _lomp_TargetTask_Setup_WithDep(0, h, _of4, fpagr4, NULL, 0, 2, taskDepArray4, 0);

    #else

      // 1
     #pragma omp target map(A, Z) depend(out: A[0]) nowait
      {
        int i, z;
        for(i=0; i<N; i++) A[i]++;
        for(z=0; z<Z[N]; z++) { for(i=0; i<N; i++) A[i] = A[i] / Z[i]; }
      }
      
      // 2
      #pragma omp target map(A, B, Z) depend(in: A[0]) depend(inout: B[0]) nowait
      {
        int i, z;
        for(i=0; i<N; i++) B[i] += A[i];
        for(z=0; z<Z[N]; z++) { for(i=0; i<N; i++) A[i] = A[i] / Z[i]; }
      }
      
      // 3
      #pragma omp target map(A, C, Z) depend(in: A[0]) depend(inout: C[0]) nowait
      {
        int i, z;
        for(i=0; i<N; i++) C[i] += A[i];
        for(z=0; z<Z[N]; z++) { for(i=0; i<N; i++) A[i] = A[i] / Z[i]; }
     }
      
      // 4
      #pragma omp target map(A, B, C, Z) depend(inout: B[0], C[0]) nowait
      {
        int i, z;
        for(i=0; i<N; i++) A[i] = B[i]+ C[i];
        for(z=0; z<Z[N]; z++) { for(i=0; i<N; i++) A[i] = A[i] / Z[i]; }
      }
    #endif
    #pragma omp taskwait
  }
    
  errors = 0;
  for(i=0; i<N; i++) {
    int a, b, c, d;
    a = i;
    b = i + 1;
    c = i + 2;
    d = i + 3;
    //
    a++;
    b += a;
    c += a;
    a = b + c;
    if (A[i] != a) printf("%d: got %d, expected %d; error %d\n", i, A[i], a, ++errors);
    if (errors>25) break;
  }
  printf("completed with %d errors\n", errors);
  return 1;   
}
