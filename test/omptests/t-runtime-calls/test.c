
#include <stdio.h>
#include <omp.h>

#include "../utilities/check.h"
#include "../utilities/utilities.h"

#define TRIALS (1)

#define N (992)

#define INIT() INIT_LOOP(N, {C[i] = 1; D[i] = i; E[i] = -i;})

#define ZERO(X) ZERO_ARRAY(N, X)

//
// TODO: Add the following runtime calls.
// omp_set_num_threads
//
// All Lock Routines.
//

int main(void) {
  // CHECK: Able to use offloading!
  check_offloading();

  int fail;
  double A[N], B[N], C[N], D[N], E[N];

  INIT();

  //
  // Test: omp_get_num_threads()
  //
  ZERO(A);
  TEST({
    A[0] = omp_get_num_threads();  // 1
  _Pragma("omp parallel num_threads(128)")
    {
      if (omp_get_thread_num() == 3) {
        A[0] += omp_get_num_threads();  // 128
      }
    }
  }, VERIFY(0, 1, A[i], 129));

  //
  // Test: omp_get_max_threads() (depends on device type)
  //
  ZERO(A);
  TEST({
      A[0] = omp_get_max_threads();
      _Pragma("omp parallel")
      {
	  if (omp_get_thread_num() == 0) {
	    A[0] += omp_get_max_threads();  // 1
	    A[1] = omp_get_num_threads();
	  }
      }
    }, if (!omp_is_initial_device()) VERIFY(0, 1, A[i], A[1] + 1));

  //
  // Test: omp_get_num_procs()
  //
  ZERO(A);
  TEST({
    A[0] = omp_get_num_procs();
  _Pragma("omp parallel")
    {
      if (omp_get_thread_num() == 18) {
        A[0] += omp_get_num_procs();
        A[1] = 2*omp_get_num_threads();
      }
    }
  }, VERIFY(0, 1, A[i], A[1]));

  //
  // Test: omp_in_parallel()
  //
  ZERO(A);
  TEST({
    A[0] = omp_in_parallel();  // 0
   // Serialized parallel
  _Pragma("omp parallel num_threads(19) if (A[0] == 1)")
    {
      A[0] += omp_in_parallel();  // 0
    }
    // Parallel execution
  _Pragma("omp parallel num_threads(19) if (A[0] == 0)")
    {
      if (omp_get_thread_num() == 18) {
        A[0] += omp_in_parallel();  // 1
      }
    }
  }, VERIFY(0, 1, A[i], 1));

  //
  // Test: omp_set/get_dynamic()
  //
  ZERO(A);
  TEST({
    A[0] = omp_get_dynamic();   // 0
    omp_set_dynamic(1);
    A[0] += omp_get_dynamic();  // 1
  _Pragma("omp parallel num_threads(19)")
    {
      if (omp_get_thread_num() == 18) {
        A[0] += omp_get_dynamic();  // 1
        omp_set_dynamic(0);  // Only for this parallel region.
      }
    }
    A[0] += omp_get_dynamic();  // 1
  }, VERIFY(0, 1, A[i], 3));

  //
  // Test: omp_get_cancellation()
  // FIXME: Rewrite test case once we have cancellation support.
  //
  ZERO(A);
  TEST({
    A[0] = omp_get_cancellation();  // 0
  _Pragma("omp parallel num_threads(19)")
    {
      if (omp_get_thread_num() == 18) {
        A[0] += omp_get_cancellation();  // 0
      }
    }
  }, VERIFY(0, 1, A[i], 0));

  //
  // Test: omp_set/get_nested().  Not used on the device currently.
  //
  ZERO(A);
  TEST({
    A[0] = omp_get_nested();   // 0
    omp_set_nested(0);
    A[0] += omp_get_nested();  // 0
  _Pragma("omp parallel num_threads(19)")
    {
      if (omp_get_thread_num() == 18) {
        A[0] += omp_get_nested();  // 0
        omp_set_nested(0);
      }
    }
    A[0] += omp_get_nested();  // 0
  }, VERIFY(0, 1, A[i], 0));

  //
  // Test: omp_set/get_schedule().
  //
  ZERO(A);
  int result = 2 * (omp_sched_static + omp_sched_dynamic + omp_sched_guided) + omp_sched_static;
  result += 2 * (1110) + 10;
  TEST({
    omp_sched_t t; int chunk_size;
    t = omp_sched_static; chunk_size = 10;
    omp_set_schedule(t, chunk_size);
    t = 0; chunk_size = 0;
    omp_get_schedule(&t, &chunk_size);
    A[0] = t + chunk_size;
    t = omp_sched_dynamic; chunk_size = 100;
    omp_set_schedule(t, chunk_size);
    t = 0; chunk_size = 0;
    omp_get_schedule(&t, &chunk_size);
    A[0] += t + chunk_size;
    t = omp_sched_guided; chunk_size = 1000;
    omp_set_schedule(t, chunk_size);
    t = 0; chunk_size = 0;
    omp_get_schedule(&t, &chunk_size);
    A[0] += t + chunk_size;
    t = omp_sched_static; chunk_size = 10;
    omp_set_schedule(t, chunk_size);
  _Pragma("omp parallel num_threads(19)")
    {
      if (omp_get_thread_num() == 18) {
        omp_sched_t t; int chunk_size;
        t = omp_sched_static; chunk_size = 10;
        omp_set_schedule(t, chunk_size);
        t = 0; chunk_size = 0;
        omp_get_schedule(&t, &chunk_size);
        A[0] += t + chunk_size;
        t = omp_sched_dynamic; chunk_size = 100;
        omp_set_schedule(t, chunk_size);
        t = 0; chunk_size = 0;
        omp_get_schedule(&t, &chunk_size);
        A[0] += t + chunk_size;
        t = omp_sched_guided; chunk_size = 1000;
        omp_set_schedule(t, chunk_size);
        t = 0; chunk_size = 0;
        omp_get_schedule(&t, &chunk_size);
        A[0] += t + chunk_size;
      }
    }
    t = 0; chunk_size = 0;
    omp_get_schedule(&t, &chunk_size);  // should read 1, 10;
    A[0] += t + chunk_size;
  }, VERIFY(0, 1, A[i], result));

  //
  // Test: omp_get_thread_limit()
  //
  ZERO(A);
  TEST({
    A[0] = omp_get_thread_limit();
  _Pragma("omp parallel")
    {
      if (omp_get_thread_num() == 0) {
        A[0] += omp_get_thread_limit();
        A[1] = 2*omp_get_num_threads();
      }
    }
  }, VERIFY(0, 1, A[i], A[1]));

  //
  // Test: omp_set/get_max_active_levels()
  //
  ZERO(A);
  TEST({
   // Our runtime ignores this.
   omp_set_max_active_levels(1);
    A[0] = omp_get_max_active_levels();  // 1
  _Pragma("omp parallel num_threads(19)")
    {
      if (omp_get_thread_num() == 18) {
        A[0] += omp_get_max_active_levels();  // 1
      }
    }
  }, VERIFY(0, 1, A[i], 2));

  //
  // Test: omp_get_level()
  //
  ZERO(A);
  TEST({
    A[0] = omp_get_level();  // 0
  _Pragma("omp parallel num_threads(19)")
    {
      if (omp_get_thread_num() == 18) {
        A[0] += omp_get_level();  // 1
      }
    }
  }, VERIFY(0, 1, A[i], 1));

  //
  // Test: omp_get_ancestor_thread_num()
  //
  ZERO(A);
  TEST({
      A[0] = omp_get_ancestor_thread_num(0);  // 0
      _Pragma("omp parallel num_threads(19)")
      {
	if (omp_get_thread_num() == 18) {
	  A[0] += omp_get_ancestor_thread_num(0) + omp_get_ancestor_thread_num(1);  // 0 + 18
	}
      }
    }, VERIFY(0, 1, A[i], 18));

  //
  // Test: omp_get_team_size()
  //
  ZERO(A);
  TEST({
    A[0] = omp_get_team_size(0) + omp_get_team_size(1);  // 1 + 1
  _Pragma("omp parallel num_threads(19)")
    {
      if (omp_get_thread_num() == 18) {
        A[0] += omp_get_team_size(0) + omp_get_team_size(1);  // 1 + 19
      }
    }
    }, if (!omp_is_initial_device()) VERIFY(0, 1, A[i], 22)); // TODO: fix host execution

  //
  // Test: omp_get_active_level()
  //
  ZERO(A);
  TEST({
    A[0] = omp_get_active_level();  // 0
  _Pragma("omp parallel num_threads(19)")
    {
      if (omp_get_thread_num() == 18) {
        A[0] += omp_get_active_level();  // 1
      }
    }
  }, VERIFY(0, 1, A[i], 1));

  //
  // Test: omp_in_final()
  //
  ZERO(A);
  TEST({
      A[0] = omp_in_final();  // 1  always returns true.
      _Pragma("omp parallel num_threads(19)")
      {
	if (omp_get_thread_num() == 18) {
	  A[0] += omp_in_final();  // 1  always returns true.
	}
      }
    }, VERIFY(0, 1, A[i], omp_is_initial_device() ? 0 : 2));

  //
  // Test: omp_get_proc_bind()
  //
  ZERO(A);
  TEST({
    A[0] = omp_get_proc_bind();  // 1  always returns omp_proc_bind_true.
  _Pragma("omp parallel num_threads(19)")
    {
      if (omp_get_thread_num() == 18) {
        A[0] += omp_get_proc_bind();  // 1  always returns omp_proc_bind_true.
      }
    }
    },  VERIFY(0, 1, A[i], omp_is_initial_device() ? 0 : 2));

#if 0
  //
  // Test: Place routines (linking only).
  //
  ZERO(A);
  TEST({
    (void) omp_get_num_places();
    (void) omp_get_place_num_procs(0);
    int *ids;
    omp_get_place_proc_ids(0, ids);
    (void) omp_get_place_num();
    (void) omp_get_partition_num_places();
    int *place_nums;
    omp_get_partition_place_nums(place_nums);
  }, VERIFY(0, 1, A[i], 0));
#endif

  //
  // Test: omp_set/get_default_device()
  //
  ZERO(A);
  TEST({
    omp_set_default_device(0); // Not used on device.

    A[0] = omp_get_default_device();  // 0  always returns 0.
  _Pragma("omp parallel num_threads(19)")
    {
      if (omp_get_thread_num() == 18) {
        A[0] += omp_get_default_device();  // 0  always returns 0.
      }
    }
  }, VERIFY(0, 1, A[i], 0));

  //
  // Test: omp_get_num_devices(). Undefined on the target.
  //
  ZERO(A);
  TEST({
    A[0] = omp_get_num_devices();
  _Pragma("omp parallel num_threads(19)")
    {
      if (omp_get_thread_num() == 18) {
        A[1] = omp_get_num_devices();
      }
    }
  }, VERIFY(0, 1, A[i], A[i] - A[1]));

  //
  // Test: omp_get_num_teams(), omp_get_team_num()
  // FIXME: Start teams region when supported.
  //
  ZERO(A);
  TEST({
    A[0] = omp_get_num_teams();  // 1
    A[0] += omp_get_team_num(); // 0
  _Pragma("omp parallel num_threads(19)")
    {
      if (omp_get_thread_num() == 18) {
        A[0] += omp_get_num_teams();  // 1
        A[0] += omp_get_team_num();   // 0
      }
    }
  }, VERIFY(0, 1, A[i], 2));

  //
  // Test: omp_is_initial_device()
  //
  ZERO(A);
  A[1] = omp_is_initial_device();
  TEST({
    A[0] = omp_is_initial_device();  // 0
  _Pragma("omp parallel num_threads(19)")
    {
      if (omp_get_thread_num() == 18) {
        A[0] += omp_is_initial_device();  // 0
      }
    }
    }, VERIFY(0, 1, A[i], omp_is_initial_device() ? A[1] - A[1] : 2.0));

  return 0;

#if 0
  //
  // Test: omp_get_initial_device(). Unspecified behavior when
  // called from device.
  //
  ZERO(A);
  TEST({
    A[0] = omp_get_initial_device();
  _Pragma("omp parallel num_threads(19)")
    {
      if (omp_get_thread_num() == 18) {
        A[0] -= omp_get_initial_device();
      }
    }
  }, VERIFY(0, 1, A[i], 0));
#endif

#if 0
  //
  // Test: omp_get_max_task_priority().
  // TODO: Not used on the gpu at the moment.
  //
  ZERO(A);
  TEST({
    A[0] = omp_get_max_task_priority();
  _Pragma("omp parallel num_threads(19)")
    {
      if (omp_get_thread_num() == 18) {
        A[0] -= omp_get_max_task_priority();
      }
    }
  }, VERIFY(0, 1, A[i], 0));
#endif


  //
  // Test: Timing Routines (linking only).
  //
  ZERO(A);
  TEST({
    double precision;
    precision = omp_get_wtick();
    double start; double end;
    start = omp_get_wtime();
    end = omp_get_wtime();
  }, VERIFY(0, 1, A[i], 0));

  return 0;
}
