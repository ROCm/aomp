// This simple test only checks that
// detach clause passes parse+sema and
// omp_fulfill_event in the detached task
// works correctly.
// Inspired by document: OpenMP API Examples
// version 5.1 August 2021
// Section 4.4. "Task Detachment"

#include <stdio.h>
#include <omp.h>

int not_done_in_detached_state = 1;

void async_work(omp_event_handle_t event) {
  printf("Doing some good (async) work!\n");
  not_done_in_detached_state = 0;
  omp_fulfill_event(event);
}

void work() {
  printf("Doing some good work!\n");
  not_done_in_detached_state = 1;
}

int main() {
  int async=1;
  #pragma omp parallel
  #pragma omp masked
  {
    omp_event_handle_t event;
    #pragma omp task detach(event)
    {
      if(async) {
	async_work(event);
      } else {
	work();
	omp_fulfill_event(event);
      }
    }
    // Other work

    #pragma omp taskwait
  }
  return not_done_in_detached_state;
}
