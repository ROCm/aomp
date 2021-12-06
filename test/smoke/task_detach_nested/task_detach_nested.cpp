#include <stdio.h>
#include <omp.h>

int test_callback(omp_event_handle_t event) {
  omp_fulfill_event(event);
  return 1;
}

int main() {
  int x = 0;
  int y = 0;

  #pragma omp target teams map(tofrom: x, y)
  {
    #pragma omp parallel
    {
      #pragma omp single
      {
	omp_event_handle_t flag_event;
	#pragma omp task depend(out: y, x) detach(flag_event)  // **task A**
	{
	  y++;
	  #pragma omp task // **task C**
	  {
	    x = 1;
	    #pragma omp flush
	    test_callback(flag_event); // **this one calls fulfill_event for task A**
	  }
	}

	#pragma omp task depend(in: y)  // **task B**
	{
	  y++;
	}
      } // end single
    }
  }

  int err = 0;
  if (x != 1 || y != 2) {
    printf("Error: x = %d (exp: 1) and y = %d (exp: 2)\n", x, y);
    err = 1;
  }
  return err;
}
