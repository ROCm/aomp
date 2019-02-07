#include <stdlib.h>
#include <stdio.h>
#include "omp.h"

#include "../utilities/check.h"

#define PRINT(_args...)
//#define PRINT(_args...) printf(_args)

#define N 64

#define TRY_TASK 1
#define TASK_COMPUTE 1


#define OFF 1
#define T1  1
#define T1  1
#define T2  1
#define T3  1
#define T4  1
#define T5  1
#define T6  1
#define T7  1
#define T8  1
#define T9  1
#define T10 1
#define T11 1
#define T12 1

int main ()
{
  int a[N], aa[N];
  int b[N], bb[N];
  int c[N], cc[N];
  int d[N], dd[N];
  int e[N], ee[N];
  int i, errors;
  int cond = 0;

#if OFF
  check_offloading();
#endif

  // Test: task within target

#if T1
  // init
  for(i=0; i<N; i++) {
    a[i] = aa[i] = i+1;
    b[i] = bb[i] = 2*i +1;
  }

  // target starts 1 team and many threads in it
  #pragma omp target map(tofrom: a, b)
  {
    int id = omp_get_thread_num();
    a[id]++;
    #if TRY_TASK
      #pragma omp task firstprivate(id) shared(b) default(none)
      {
        #if TASK_COMPUTE
          PRINT("hi alex from %d\n", id);
          b[id]++;
        #endif
      }
      #pragma omp taskwait
    #endif
  }
  // reproduce
  aa[0]++;
  #if TRY_TASK && TASK_COMPUTE
    bb[0]++;
  #endif
  // verify
  errors = 0;
  for(i=0; i<N; i++) {
    if (a[i] != aa[i]) printf("%4i: got a %d, expected %d, error %d\n", i, a[i], aa[i], ++errors);
    if (b[i] != bb[i]) printf("%4i: got b %d, expected %d, error %d\n", i, b[i], bb[i], ++errors);
  }
  printf("#1 got %d errors\n", errors);
#endif

  // Test: task within parallel

#if T2
  // init
  for(i=0; i<N; i++) {
    a[i] = aa[i] = i+1;
    b[i] = bb[i] = 2*i +1;
  }

  // target starts 1 team and many threads in it
  #pragma omp target map(tofrom: a, b)
  {
    #pragma omp parallel num_threads(64)
    {
      int id = omp_get_thread_num();
      a[id]++;
      #if TRY_TASK
        #pragma omp task firstprivate(id) shared(b)
        {
          #if TASK_COMPUTE
            PRINT("hi alex from  %d\n", id);
            b[id]++;
          #endif
        }
      #endif
    }
  }
  // reproduce
  for(i=0; i<N; i++) {
    aa[i]++;
    #if TRY_TASK && TASK_COMPUTE
      bb[i]++;
    #endif
  }

  // verify
  errors = 0;
  for(i=0; i<N; i++) {
    if (a[i] != aa[i]) printf("%4i: got a %d, expected %d, error %d\n", i, a[i], aa[i], ++errors);
    if (b[i] != bb[i]) printf("%4i: got b %d, expected %d, error %d\n", i, b[i], bb[i], ++errors);
  }
  printf("#2 got %d errors\n", errors);
#endif

  // Test: multiple nested tasks in parallel region

#if T3
    // init
  for(i=0; i<N; i++) {
    a[i] = aa[i] = i+1;
    b[i] = bb[i] = 2*i +1;
  }

  // target starts 1 team and many threads in it
  #pragma omp target map(tofrom: a, b)
  {
    #pragma omp parallel num_threads(64)
    {
      int id = omp_get_thread_num();
      a[id]++;
      #pragma omp task firstprivate(id) shared(b)
      {
        PRINT("hi alex from  %d\n", id);
        #pragma omp atomic
        b[id]++;

        #pragma omp task firstprivate(id) shared(b)
        {
          PRINT("hi alex from  %d\n", id);
          #pragma omp atomic
          b[id]++;

          #pragma omp task firstprivate(id) shared(b)
          {
            PRINT("hi alex from  %d\n", id);
            #pragma omp atomic
            b[id]++;
          }
        }
      }
    }
  }

  // reproduce
  for(i=0; i<N; i++) {
    aa[i]++;
    bb[i]+=3;
  }

  // verify
  errors = 0;
  for(i=0; i<N; i++) {
    if (a[i] != aa[i]) printf("%4i: got a %d, expected %d, error %d\n", i, a[i], aa[i], ++errors);
    if (b[i] != bb[i]) printf("%4i: got b %d, expected %d, error %d\n", i, b[i], bb[i], ++errors);
  }
  printf("#3 got %d errors\n", errors);
#endif

  // Test: three successive tasks in a parallel region

#if T4
  // init
  for(i=0; i<N; i++) {
    a[i] = aa[i] = i+1;
    b[i] = bb[i] = 2*i +1;
  }

  // target starts 1 team and many threads in it
  #pragma omp target map(tofrom: a, b)
  {
    #pragma omp parallel num_threads(64)
    {
      int id = omp_get_thread_num();
      a[id]++;
      #pragma omp task firstprivate(id) shared(b)
      {
        PRINT("hi alex from %d\n", id);
        #pragma omp atomic
        b[id]++;
      }
      #pragma omp task firstprivate(id) shared(b)
      {
        PRINT("hi alex from %d\n", id);
        #pragma omp atomic
        b[id]++;
      }
      #pragma omp task firstprivate(id) shared(b)
      {
        PRINT("hi alex from %d\n", id);
        #pragma omp atomic
        b[id]++;
      }
    }
  }
  // reproduce
  for(i=0; i<N; i++) {
    aa[i]++;
    bb[i]+=3;
  }

  // verify
  errors = 0;
  for(i=0; i<N; i++) {
    if (a[i] != aa[i]) printf("%4i: got a %d, expected %d, error %d\n", i, a[i], aa[i], ++errors);
    if (b[i] != bb[i]) printf("%4i: got b %d, expected %d, error %d\n", i, b[i], bb[i], ++errors);
  }
  printf("#4 got %d errors\n", errors);
#endif

  // Test: change of context when entering/exiting tasks

#if T5
  // init
  for(i=0; i<N; i++) {
    a[i] = aa[i] = i+1;
    b[i] = bb[i] = 2*i+1;
    c[i] = cc[i] = 3*i+1;
    d[i] = dd[i] = 4*i+1;
    e[i] = ee[i] = 5*i+1;
  }

  // target starts 1 team and many threads in it
  #pragma omp target map(tofrom: a, b, c, d, e)
  {
    omp_set_schedule(omp_sched_static, 1);
    #pragma omp parallel num_threads(64)
    {
      omp_set_schedule(omp_sched_static, 2);
      int id = omp_get_thread_num();
      // task 1
      #pragma omp task firstprivate(id) shared(b, c, d, e)
      {
        omp_set_schedule(omp_sched_static, 3);
        PRINT("hi alex from %d\n", id);
        // task 2
        #pragma omp task firstprivate(id) shared(b, c, d, e)
        {
          omp_set_schedule(omp_sched_static, 4);
          PRINT("hi alex from %d\n", id);
          // task 3
          #pragma omp task firstprivate(id) shared(b, c, d, e)
          {
            omp_set_schedule(omp_sched_static, 5);
            PRINT("hi alex from %d\n", id);
            // task 3
            omp_sched_t s; int chunk;
            omp_get_schedule(&s, &chunk);
            if (s == omp_sched_static && chunk == 5) e[id]++;
          }
          // task 2
          omp_sched_t s; int chunk;
          omp_get_schedule(&s, &chunk);
          if (s == omp_sched_static && chunk == 4) d[id]++;
        }
        // task 1
        omp_sched_t s; int chunk;
        omp_get_schedule(&s, &chunk);
        if (s == omp_sched_static && chunk == 3) c[id]++;
      }
      // par
      omp_sched_t s; int chunk;
      omp_get_schedule(&s, &chunk);
      if (s == omp_sched_static && chunk == 2) b[id]++;
    }
    // team
    omp_sched_t s; int chunk;
    omp_get_schedule(&s, &chunk);
    if (s == omp_sched_static && chunk == 1) a[0]++;
  }
  // reproduce
  aa[0]++;
  for(i=0; i<N; i++) {
    bb[i]++;
    cc[i]++;
    dd[i]++;
    ee[i]++;
  }

  // verify
  errors = 0;
  for(i=0; i<N; i++) {
    if (a[i] != aa[i]) printf("%4i: got a %d, expected %d, error %d\n", i, a[i], aa[i], ++errors);
    if (b[i] != bb[i]) printf("%4i: got b %d, expected %d, error %d\n", i, b[i], bb[i], ++errors);
    if (c[i] != cc[i]) printf("%4i: got c %d, expected %d, error %d\n", i, c[i], cc[i], ++errors);
    if (d[i] != dd[i]) printf("%4i: got d %d, expected %d, error %d\n", i, d[i], dd[i], ++errors);
    if (e[i] != ee[i]) printf("%4i: got e %d, expected %d, error %d\n", i, e[i], ee[i], ++errors);
  }

  printf("#5 got %d errors\n", errors);
#endif

  // Test: change of context when using if clause

#if T6
  // init
  for(i=0; i<N; i++) {
    a[i] = aa[i] = i+1;
    b[i] = bb[i] = 2*i+1;
    c[i] = cc[i] = 3*i+1;
    d[i] = dd[i] = 4*i+1;
    e[i] = ee[i] = 5*i+1;
  }

  // target starts 1 team and many threads in it
  #pragma omp target map(tofrom: a, b, c, d, e, cond)
  {
    omp_set_schedule(omp_sched_static, 1);
    #pragma omp parallel num_threads(64)
    {
      omp_set_schedule(omp_sched_static, 2);
      int id = omp_get_thread_num();
      // task 1
      #pragma omp task firstprivate(id) shared(b, c, d, e) if(cond)
      {
        omp_set_schedule(omp_sched_static, 3);
        PRINT("hi alex from  %d\n", id);
        // task 2
        #pragma omp task firstprivate(id) shared(b, c, d, e) if(cond)
        {
          omp_set_schedule(omp_sched_static, 4);
          PRINT("hi alex from  %d\n", id);
          // task 3
          #pragma omp task firstprivate(id) shared(b, c, d, e) if(cond)
          {
            omp_set_schedule(omp_sched_static, 5);
            PRINT("hi alex from  %d\n", id);
            // task 3
            omp_sched_t s; int chunk;
            omp_get_schedule(&s, &chunk);
            if (s == omp_sched_static && chunk == 5) e[id]++;
          }
          // task 2
          omp_sched_t s; int chunk;
          omp_get_schedule(&s, &chunk);
          if (s == omp_sched_static && chunk == 4) d[id]++;
        }
        // task 1
        omp_sched_t s; int chunk;
        omp_get_schedule(&s, &chunk);
        if (s == omp_sched_static && chunk == 3) c[id]++;
      }
      // par
      omp_sched_t s; int chunk;
      omp_get_schedule(&s, &chunk);
      if (s == omp_sched_static && chunk == 2) b[id]++;
    }
    // team
    omp_sched_t s; int chunk;
    omp_get_schedule(&s, &chunk);
    if (s == omp_sched_static && chunk == 1) a[0]++;
  }
  // reproduce
  aa[0]++;
  for(i=0; i<N; i++) {
    bb[i]++;
    cc[i]++;
    dd[i]++;
    ee[i]++;
  }

  // verify
  errors = 0;
  for(i=0; i<N; i++) {
    if (a[i] != aa[i]) printf("%4i: got a %d, expected %d, error %d\n", i, a[i], aa[i], ++errors);
    if (b[i] != bb[i]) printf("%4i: got b %d, expected %d, error %d\n", i, b[i], bb[i], ++errors);
    if (c[i] != cc[i]) printf("%4i: got c %d, expected %d, error %d\n", i, c[i], cc[i], ++errors);
    if (d[i] != dd[i]) printf("%4i: got d %d, expected %d, error %d\n", i, d[i], dd[i], ++errors);
    if (e[i] != ee[i]) printf("%4i: got e %d, expected %d, error %d\n", i, e[i], ee[i], ++errors);
  }

  printf("#6 got %d errors\n", errors);
#endif

  // Test: final

#if T7
  // init
  for(i=0; i<N; i++) {
    a[i] = aa[i] = i+1;
    b[i] = bb[i] = 2*i +1;
  }

  // target starts 1 team and many threads in it
  #pragma omp target map(tofrom: a, b)
  {
    #pragma omp parallel num_threads(64)
    {
      int id = omp_get_thread_num();
      a[id]++;
      #pragma omp task firstprivate(id) shared(b) final(1)
      {
        PRINT("hi alex from  %d\n", id);
        #pragma omp atomic
        b[id]++;

        #pragma omp task firstprivate(id) shared(b) final(1)
        {
          PRINT("hi alex from  %d\n", id);
          #pragma omp atomic
          b[id]++;

          #pragma omp task firstprivate(id) shared(b) final(1)
          {
            PRINT("hi alex from  %d\n", id);
            #pragma omp atomic
            b[id]++;
          }
        }
      }
    }
  }

  // reproduce
  for(i=0; i<N; i++) {
    aa[i]++;
    bb[i]+=3;
  }

  // verify
  errors = 0;
  for(i=0; i<N; i++) {
    if (a[i] != aa[i]) printf("%4i: got a %d, expected %d, error %d\n", i, a[i], aa[i], ++errors);
    if (b[i] != bb[i]) printf("%4i: got b %d, expected %d, error %d\n", i, b[i], bb[i], ++errors);
  }
  printf("#7 got %d errors\n", errors);
#endif

  // Test: untied

#if T8 && 0
  // init
  for(i=0; i<N; i++) {
    a[i] = aa[i] = i+1;
    b[i] = bb[i] = 2*i +1;
  }

  // target starts 1 team and many threads in it
  #pragma omp target map(tofrom: a, b)
  {
    #pragma omp parallel num_threads(64)
    {
      int id = omp_get_thread_num();
      a[id]++;
      #pragma omp task firstprivate(id) shared(b) untied
      {
        PRINT("hi alex from  %d\n", id);
        #pragma omp atomic
        b[id]++;

        #pragma omp task firstprivate(id) shared(b) untied
        {
          PRINT("hi alex from  %d\n", id);
          #pragma omp atomic
          b[id]++;

          #pragma omp task firstprivate(id) shared(b) untied
          {
            PRINT("hi alex from  %d\n", id);
            #pragma omp atomic
            b[id]++;
          }
        }
      }
    }
  }

  // reproduce
  for(i=0; i<N; i++) {
    aa[i]++;
    bb[i]+=3;
  }

  // verify
  errors = 0;
  for(i=0; i<N; i++) {
    if (a[i] != aa[i]) printf("%4i: got a %d, expected %d, error %d\n", i, a[i], aa[i], ++errors);
    if (b[i] != bb[i]) printf("%4i: got b %d, expected %d, error %d\n", i, b[i], bb[i], ++errors);
  }
  printf("#8 got %d errors\n", errors);
#endif

  // Test: mergeaeble

#if T9
  // init
  for(i=0; i<N; i++) {
    a[i] = aa[i] = i+1;
    b[i] = bb[i] = 2*i +1;
  }

  // target starts 1 team and many threads in it
  #pragma omp target map(tofrom: a, b)
  {
    #pragma omp parallel num_threads(64)
    {
      int id = omp_get_thread_num();
      a[id]++;
      #pragma omp task firstprivate(id) shared(b)
      {
        PRINT("hi alex from  %d\n", id);
        //#pragma omp atomic
        b[id]++;

        #pragma omp task firstprivate(id) shared(b) mergeable
        {
          PRINT("hi alex from  %d\n", id);
          //#pragma omp atomic
          b[id]++;

          #pragma omp task firstprivate(id) shared(b) mergeable
          {
            PRINT("hi alex from  %d\n", id);
            //#pragma omp atomic
            b[id]++;
          }
        }
      }
    }
  }

  // reproduce
  for(i=0; i<N; i++) {
    aa[i]++;
    bb[i]+=3;
  }

  // verify
  errors = 0;
  for(i=0; i<N; i++) {
    if (a[i] != aa[i]) printf("%4i: got a %d, expected %d, error %d\n", i, a[i], aa[i], ++errors);
    if (b[i] != bb[i]) printf("%4i: got b %d, expected %d, error %d\n", i, b[i], bb[i], ++errors);
  }
  printf("#9 got %d errors\n", errors);
#endif

  // Test: private

#if T10 && 0
  /*
    Test disabled because this test only works on the GPU, where a
    task is guaranteed to work on the same thread as it is
    created. This is not true in general on the host. So we cannot use
    this test generally. Thus I am disabling it here
   */

  // init
  for(i=0; i<N; i++) {
    a[i] = aa[i] = i+1;
    b[i] = bb[i] = 2*i +1;
  }

  // target starts 1 team and many threads in it
  #pragma omp target map(tofrom: a, b)
  {
    #pragma omp parallel num_threads(64)
    {
      int id = omp_get_thread_num();
      a[id]++;
      #if TRY_TASK
        #pragma omp task private(id) shared(b)
        {
          int id = omp_get_thread_num();
          #if TASK_COMPUTE
            PRINT("hi alex from  %d\n", id);
            b[id]++;
          #endif
        }
      #endif
    }
  }
  // reproduce
  for(i=0; i<N; i++) {
    aa[i]++;
    #if TRY_TASK && TASK_COMPUTE
      bb[i]++;
    #endif
  }

  // verify
  errors = 0;
  for(i=0; i<N; i++) {
    if (a[i] != aa[i]) printf("%4i: got a %d, expected %d, error %d\n", i, a[i], aa[i], ++errors);
    if (b[i] != bb[i]) printf("%4i: got b %d, expected %d, error %d\n", i, b[i], bb[i], ++errors);
  }
  printf("#10 got %d errors\n", errors);
#endif

  // Test: depend

#if T11
  // init
  int x;
  for(i=0; i<N; i++) {
    a[i] = aa[i] = i+1;
    b[i] = bb[i] = 2*i +1;
  }

  // target starts 1 team and many threads in it
  #pragma omp target map(tofrom: a, b)
  {
    #pragma omp parallel num_threads(64)
    {
      int id = omp_get_thread_num();
      a[id]++;
      #pragma omp task firstprivate(id) shared(b) depend(out:a[id])
      {
        PRINT("hi alex from %d\n", id);
        b[id]++;
      }
      #pragma omp task firstprivate(id) shared(b) depend(inout:a[id])
      {
        PRINT("hi alex from %d\n", id);
        b[id]++;
      }
      #pragma omp task firstprivate(id) shared(b) depend(in:a[id])
      {
        PRINT("hi alex from %d\n", id);
        b[id]++;
      }
    }
  }
  // reproduce
  for(i=0; i<N; i++) {
    aa[i]++;
    bb[i]+=3;
  }

  // verify
  errors = 0;
  for(i=0; i<N; i++) {
    if (a[i] != aa[i]) printf("%4i: got a %d, expected %d, error %d\n", i, a[i], aa[i], ++errors);
    if (b[i] != bb[i]) printf("%4i: got b %d, expected %d, error %d\n", i, b[i], bb[i], ++errors);
  }
  printf("#11 got %d errors\n", errors);
#endif

  // Test: inverted priority

#if T12
  // init
  for(i=0; i<N; i++) {
    a[i] = aa[i] = i+1;
    b[i] = bb[i] = 2*i +1;
  }

  // target starts 1 team and many threads in it
  #pragma omp target map(tofrom: a, b)
  {
    #pragma omp parallel num_threads(64)
    {
      int id = omp_get_thread_num();
      a[id]++;
      #pragma omp task firstprivate(id) shared(b) priority(0)
      {
        PRINT("hi alex from %d\n", id);
        #pragma omp atomic
        b[id]++;
      }
      #pragma omp task firstprivate(id) shared(b) priority(10)
      {
        PRINT("hi alex from %d\n", id);
        #pragma omp atomic
        b[id]++;
      }
      #pragma omp task firstprivate(id) shared(b) priority(20)
      {
        PRINT("hi alex from %d\n", id);
        #pragma omp atomic
        b[id]++;
      }
    }
  }
  // reproduce
  for(i=0; i<N; i++) {
    aa[i]++;
    bb[i]+=3;
  }

  // verify
  errors = 0;
  for(i=0; i<N; i++) {
    if (a[i] != aa[i]) printf("%4i: got a %d, expected %d, error %d\n", i, a[i], aa[i], ++errors);
    if (b[i] != bb[i]) printf("%4i: got b %d, expected %d, error %d\n", i, b[i], bb[i], ++errors);
  }
  printf("#12 got %d errors\n", errors);
#endif

  return 0;
}
