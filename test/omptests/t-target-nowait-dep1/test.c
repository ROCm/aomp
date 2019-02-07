#include <omp.h>
#include <stdio.h>

#define N 1024

#define EXPLICIT_TARGET_TASK 0

int A[N];
int B[N];
int C[N];

#if EXPLICIT_TARGET_TASK
  #define LOMP_TASK_DEP_40 1
  #define LOMP_TARGET_40 1
  #define LOMP_PROC_BIND_40 1 
  #define LOMP_OS_LINUX 1
  #define LOMP_CANCEL_40 1
  #include "/gsa/yktgsa-h1/00/eichen/new-tlomp/lomp/include/omp_interface.h"
 
  #define MAP_ITEM_SIZE (sizeof(int64_t))
  #define MAP_SIZE (4*MAP_ITEM_SIZE)


  void _of1(lomp_Handle handle, char *fparg, char *sharg)
  {
    // for now use a compiler generated target because I am not sure how to 
    // generate taret code, not sure its even possible directly from the 
    // compiler
    #if 0
      // get map pointers into frist private
      int mapNum1 = 1;
      int64_t *args1= (int64_t *) fparg;
      int64_t *args_base1 = args1 + mapNum1;
      int64_t *arg_sizes1 = args_base1 + mapNum1;
      int64_t *arg_types1 = arg_sizes1 + mapNum1;
    #endif
    #pragma omp target nowait map(A)
    {
      int i;
      for(int i=0; i<N; i++) A[i]++;
    }
  }

#endif

int main() 
{
  int i, errors;

  for(i=0; i<N; i++) {
    A[i] = i;
  }
    
  #if EXPLICIT_TARGET_TASK

    #pragma omp target data map(A) 
    {
      lomp_Handle h = _lomp_GetHandle();
      
      // first task
      lomp_TaskDep_Public *taskDepArray1;
      int mapNum1 = 1;
      void *fpagr1 = _lomp_Task_AllocateFirstPrivate_WithDeps(h, mapNum1 * MAP_SIZE, 1, &taskDepArray1);
      #if 0
        // get map pointers into frist private
        int64_t *args1= (int64_t *) fparg1;
        int64_t *args_base1 = args1 + mapNum1;
        int64_t *arg_sizes1 = args_base1 + mapNum1;
        int64_t *arg_types1 = arg_sizes1 + mapNum1;
        // define maps
        args1[0] = args_base1[0] = &A[0];
        arg_sizes1[0] = N * sizeof(int);
        arg_types1[0] = lomp_tmap_tofrom | lomp_tmap_target_param;
      #endif
      // set dependences
      taskDepArray1[0].addr = &A[0];
      taskDepArray1[0].status = LOMP_TASK_DEP_STATUS_OUT;
      // launch target task
      _lomp_TargetTask_Setup_WithDep(0, h, _of1, fpagr1, NULL, 0, 1, taskDepArray1, 0);
      
      // second task
      lomp_TaskDep_Public *taskDepArray2;
      int mapNum2 = 1;
      void *fpagr2 = _lomp_Task_AllocateFirstPrivate_WithDeps(h, mapNum2 * MAP_SIZE, 1, &taskDepArray2);
      #if 0
        // get map pointers into frist private
        int64_t *arg2s= (int64_t *) fparg2;
        int64_t *args_base2 = args2 + mapNum2;
        int64_t *arg_sizes2 = args_base2 + mapNum2;
        int64_t *arg_types2 = arg_sizes2 + mapNum2;
        // define maps
        args2[0] = args_base2[0] = &A[0];
        arg_sizes2[0] = N * sizeof(int);
        arg_types2[0] = lomp_tmap_tofrom | lomp_tmap_target_param;
      #endif
      // set dependences
      taskDepArray2[0].addr = &A[0];
      taskDepArray2[0].status = LOMP_TASK_DEP_STATUS_OUT;
      // launch target task
      _lomp_TargetTask_Setup_WithDep(0, h, _of1, fpagr2, NULL, 0, 1, taskDepArray2, 0);

      #pragma omp taskwait

    }

  #elif 1

    #pragma omp target data map(A) 
    {    
      #pragma omp target map(A) depend(out: A[0]) nowait
      {
        int i;
        for(int i=0; i<N; i++) A[i]++;
      }
      
      #pragma omp target map(A) depend(out: A[0]) nowait
      {
        int i;
        for(int i=0; i<N; i++) A[i]++;
      }
      
      #pragma omp taskwait
    }
    
  #else

      #pragma omp target enter data map(to: A) depend(out: A[0]) nowait
    
      #pragma omp target map(A) depend(out: A[0]) nowait
      {
        int i;
        for(int i=0; i<N; i++) A[i]++;
      }
      
      #pragma omp target map(A) depend(out: A[0]) nowait
      {
        int i;
        for(int i=0; i<N; i++) A[i]++;
      }

      #pragma omp target exit data map(from: A) depend(out: A[0]) nowait

      
      #pragma omp taskwait

  #endif


  errors = 0;
  for(i=0; i<N; i++) {
    if (A[i] != i+2) printf("%d: got %d, expected %d; error %d\n", i, A[i], i+2, ++errors);
    if (errors>25) break;
  }
  printf("completed with %d errors\n", errors);
  return 1;   
}
