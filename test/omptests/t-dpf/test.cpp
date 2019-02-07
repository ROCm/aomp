#include <omp.h>
#include <stdio.h>

#define MAX_N 25000

void reset_input(double *a, double *a_h, double *b, double *c) {
  for(int i = 0 ; i < MAX_N ; i++) {
    a[i] = a_h[i] = i;
    b[i] = i*2;
    c[i] = i-3;
  }
}

int main(int argc, char *argv[]) {
  double * a = (double *) malloc(MAX_N * sizeof(double));
  double * a_h = (double *) malloc(MAX_N * sizeof(double));
  double * b = (double *) malloc(MAX_N * sizeof(double));
  double * c = (double *) malloc(MAX_N * sizeof(double));

#pragma omp target enter data map(to:a[:MAX_N],b[:MAX_N],c[:MAX_N])

  // 1. no schedule clauses
  printf("no schedule clauses\n");
  for (int n = 32 ; n < MAX_N ; n+=5000) {

    reset_input(a, a_h, b, c);

#pragma omp target update to(a[:n],b[:n],c[:n])
    int t = 0;
    for (int tms = 1 ; tms <= 256 ; tms *= 2) { // 8 times
      for (int ths = 32 ; ths <= 1024 ; ths *= 2) { // 6 times
	t++;
#pragma omp target
#pragma omp teams num_teams(tms) thread_limit(ths)
	{
#pragma omp distribute parallel for
	  for (int i = 0; i < n; ++i) {
	    a[i] += b[i] + c[i];
	  }
	}
      } // loop over 'ths'
    } // loop over 'tms'

    // check results for each 'n'
    for (int times = 0 ; times < t ; times++)
      for (int i = 0; i < n; ++i)
	a_h[i] += b[i] + c[i];

#pragma omp target update from(a[:n])

    for (int i = 0; i < n; ++i) {
      if (a_h[i] != a[i]) {
	printf("Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, a_h[i], a[i]);
	return 1;
      }
    }
  } // loop over 'n'
  printf("Succeeded\n");

  // 2. schedule static no chunk
  printf("schedule static no chunk\n");
  for (int n = 32 ; n < MAX_N ; n+=5000) {

    reset_input(a, a_h, b, c);

#pragma omp target update to(a[:n],b[:n],c[:n])
    int t = 0;
    for (int tms = 1 ; tms <= 256 ; tms *= 2) { // 8 times
      for (int ths = 32 ; ths <= 1024 ; ths *= 2) { // 6 times
	t++;
#pragma omp target
#pragma omp teams num_teams(tms) thread_limit(ths)
	{
#pragma omp distribute parallel for schedule(static)
	  for (int i = 0; i < n; ++i) {
	    a[i] += b[i] + c[i];
	  }
	}
      } // loop over 'ths'
    } // loop over 'tms'

    // check results for each 'n'
    for (int times = 0 ; times < t ; times++)
      for (int i = 0; i < n; ++i)
	a_h[i] += b[i] + c[i];

#pragma omp target update from(a[:n])

    for (int i = 0; i < n; ++i) {
      if (a_h[i] != a[i]) {
	printf("Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, a_h[i], a[i]);
	return 1;
      }
    }
  } // loop over 'n'
  printf("Succeeded\n");

  // 3. schedule static chunk
  printf("schedule static chunk\n");
  for (int n = 32 ; n < MAX_N ; n+=5000) {
    int t = 0;
    reset_input(a, a_h, b, c);

#pragma omp target update to(a[:n],b[:n],c[:n])

    for (int tms = 1 ; tms <= 256 ; tms *= 2) { // 8 times
      for (int ths = 32 ; ths <= 1024 ; ths *= 2) { // 6 times
	for(int sch = 1 ; sch <= n ; sch *= 3000) {
	  t++;
#pragma omp target
#pragma omp teams num_teams(tms) thread_limit(ths)
	  {
#pragma omp distribute parallel for schedule(static,sch)
	    for (int i = 0; i < n; ++i) {
	      a[i] += b[i] + c[i];
	    }
	  }
	} // loop 'sch'
      } // loop 'ths'
    } // loop 'tms'

    // check results for each 'n'
    for (int times = 0 ; times < t ; times++)
      for (int i = 0; i < n; ++i)
	a_h[i] += b[i] + c[i];

#pragma omp target update from(a[:n])

    for (int i = 0; i < n; ++i) {
      if (a_h[i] != a[i]) {
	printf("Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, a_h[i], a[i]);
	return 1;
      }
    }
  } // loop 'n'
  printf("Succeeded\n");

  // 4. schedule dynamic no chunk (debugging)
  printf("schedule dynamic no chunk\n");
  for (int n = 32 ; n < MAX_N ; n+=5000) {
    int t = 0;
    reset_input(a, a_h, b, c);

#pragma omp target update to(a[:n],b[:n],c[:n])

    for (int tms = 1 ; tms <= 256 ; tms *= 2) { // 8 times
      for (int ths = 32 ; ths <= 1024 ; ths *= 2) { // 6 times
	  t++;
#pragma omp target
#pragma omp teams num_teams(tms) thread_limit(ths)
	  {
#pragma omp distribute parallel for schedule(dynamic)
	    for (int i = 0; i < n; ++i) {
	      a[i] += b[i] + c[i];
	    }
	  }
      } // loop 'ths'
    } // loop 'tms'


    // check results for each 'n'
    for (int times = 0 ; times < t ; times++)
      for (int i = 0; i < n; ++i)
	a_h[i] += b[i] + c[i];

#pragma omp target update from(a[:n])

    for (int i = 0; i < n; ++i) {
      if (a_h[i] != a[i]) {
	printf("Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, a_h[i], a[i]);
	return 1;
      }
    }
  } // loop 'n'
  printf("Succeeded\n");


  // 5. schedule dynamic chunk (debugging)
  printf("schedule dynamic chunk\n");
  for (int n = 32 ; n < MAX_N ; n+=5000) {
    int t = 0;
    reset_input(a, a_h, b, c);

#pragma omp target update to(a[:n],b[:n],c[:n])

    for (int tms = 1 ; tms <= 256 ; tms *= 2) { // 8 times
      for (int ths = 32 ; ths <= 1024 ; ths *= 2) { // 6 times
	for(int sch = 1 ; sch <= n ; sch *= 1200) {
	  t++;
#pragma omp target
#pragma omp teams num_teams(tms) thread_limit(ths)
	  {
#pragma omp distribute parallel for schedule(dynamic, sch)
	    for (int i = 0; i < n; ++i) {
	      a[i] += b[i] + c[i];
	    }
	  }
	} // loop 'sch'
      } // loop 'ths'
    } // loop 'tms'


    // check results for each 'n'
    for (int times = 0 ; times < t ; times++)
      for (int i = 0; i < n; ++i)
	a_h[i] += b[i] + c[i];

#pragma omp target update from(a[:n])

    for (int i = 0; i < n; ++i) {
      if (a_h[i] != a[i]) {
	printf("Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, a_h[i], a[i]);
	return 1;
      }
    }
  } // loop 'n'
  printf("Succeeded\n");

  // 6. dist_schedule static no chunk
  printf("dist_schedule static no chunk\n");
  for (int n = 32 ; n < MAX_N ; n+=5000) {

    reset_input(a, a_h, b, c);

#pragma omp target update to(a[:n],b[:n],c[:n])
    int t = 0;
    for (int tms = 1 ; tms <= 256 ; tms *= 2) { // 8 times
      for (int ths = 32 ; ths <= 1024 ; ths *= 2) { // 6 times
	t++;
#pragma omp target
#pragma omp teams num_teams(tms) thread_limit(ths)
	{
#pragma omp distribute parallel for dist_schedule(static)
	  for (int i = 0; i < n; ++i) {
	    a[i] += b[i] + c[i];
	  }
	}
      }
    }

    // check results for each 'n'
    for (int times = 0 ; times < t ; times++)
      for (int i = 0; i < n; ++i)
	a_h[i] += b[i] + c[i];

#pragma omp target update from(a[:n])

    for (int i = 0; i < n; ++i) {
      if (a_h[i] != a[i]) {
	printf("Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, a_h[i], a[i]);
	return 1;
      }
    }
  } // loop over 'n'
  printf("Succeeded\n");

  // 7. dist_schedule static chunk
  printf("dist_schedule static chunk\n");
  for (int n = 32 ; n < MAX_N ; n+=5000) {
    int t = 0;
    reset_input(a, a_h, b, c);

#pragma omp target update to(a[:n],b[:n],c[:n])

    for (int tms = 1 ; tms <= 256 ; tms *= 2) { // 8 times
      for (int ths = 32 ; ths <= 1024 ; ths *= 2) { // 6 times
	for(int sch = 128 ; sch <= n ; sch *= 10000) {
	  t++;
#pragma omp target
#pragma omp teams num_teams(tms) thread_limit(ths)
	  {
#pragma omp distribute parallel for dist_schedule(static,sch)
	    for (int i = 0; i < n; ++i) {
	      a[i] += b[i] + c[i];
	    }
	  }
	} // loop 'sch'
      } // loop 'ths'
    } // loop 'tms'

    // check results for each 'n'
    for (int times = 0 ; times < t ; times++)
      for (int i = 0; i < n; ++i)
	a_h[i] += b[i] + c[i];

#pragma omp target update from(a[:n])

    for (int i = 0; i < n; ++i) {
      if (a_h[i] != a[i]) {
	printf("Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, a_h[i], a[i]);
	return 1;
      }
    }
  } // loop 'n'
  printf("Succeeded\n");

  // 8. dist_schedule static no chunk, schedule static no chunk
  printf("dist_schedule static no chunk, schedule static no chunk\n");
  for (int n = 32 ; n < MAX_N ; n+=5000) {

    reset_input(a, a_h, b, c);

#pragma omp target update to(a[:n],b[:n],c[:n])
    int t = 0;
    for (int tms = 1 ; tms <= 256 ; tms *= 2) { // 8 times
      for (int ths = 32 ; ths <= 1024 ; ths *= 2) { // 6 times
	t++;
#pragma omp target
#pragma omp teams num_teams(tms) thread_limit(ths)
	{
#pragma omp distribute parallel for dist_schedule(static) schedule(static)
	  for (int i = 0; i < n; ++i) {
	    a[i] += b[i] + c[i];
	  }
	}
      }
    }

    // check results for each 'n'
    for (int times = 0 ; times < t ; times++)
      for (int i = 0; i < n; ++i)
	a_h[i] += b[i] + c[i];

#pragma omp target update from(a[:n])

    for (int i = 0; i < n; ++i) {
      if (a_h[i] != a[i]) {
	printf("Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, a_h[i], a[i]);
	return 1;
      }
    }
  } // loop over 'n'
  printf("Succeeded\n");

  // 9. dist_schedule static no chunk, schedule static chunk
  printf("dist_schedule static no chunk, schedule static chunk\n");
  for (int n = 32 ; n < MAX_N ; n+=5000) {
    int t = 0;
    reset_input(a, a_h, b, c);

#pragma omp target update to(a[:n],b[:n],c[:n])

    for (int tms = 1 ; tms <= 256 ; tms *= 2) { // 8 times
      for (int ths = 32 ; ths <= 1024 ; ths *= 2) { // 6 times
	for(int sch = 1 ; sch <= n ; sch *= 1000) { // speed up very slow tests
	  t++;
#pragma omp target
#pragma omp teams num_teams(tms) thread_limit(ths)
	  {
#pragma omp distribute parallel for dist_schedule(static) schedule(static,sch)
	    for (int i = 0; i < n; ++i) {
	      a[i] += b[i] + c[i];
	    }
	  }
	} // loop 'sch'
      } // loop 'ths'
    } // loop 'tms'

    // check results for each 'n'
    for (int times = 0 ; times < t ; times++)
      for (int i = 0; i < n; ++i)
	a_h[i] += b[i] + c[i];

#pragma omp target update from(a[:n])

    for (int i = 0; i < n; ++i) {
      if (a_h[i] != a[i]) {
	printf("Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, a_h[i], a[i]);
	return 1;
      }
    }
  } // loop 'n'
  printf("Succeeded\n");

  // 10. dist_schedule static chunk, schedule static no chunk
  printf("dist_schedule static chunk, schedule static no chunk\n");
  for (int n = 32 ; n < MAX_N ; n+=5000) {
    int t = 0;
    reset_input(a, a_h, b, c);

#pragma omp target update to(a[:n],b[:n],c[:n])

    for (int tms = 1 ; tms <= 256 ; tms *= 2) { // 8 times
      for (int ths = 32 ; ths <= 1024 ; ths *= 2) { // 6 times
	for(int sch = 128 ; sch <= n ; sch *= 1200) {
	  t++;
#pragma omp target
#pragma omp teams num_teams(tms) thread_limit(ths)
	  {
#pragma omp distribute parallel for dist_schedule(static,sch) schedule(static)
	    for (int i = 0; i < n; ++i) {
	      a[i] += b[i] + c[i];
	    }
	  }
	} // loop 'sch'
      } // loop 'ths'
    } // loop 'tms'

    // check results for each 'n'
    for (int times = 0 ; times < t ; times++)
      for (int i = 0; i < n; ++i)
	a_h[i] += b[i] + c[i];

#pragma omp target update from(a[:n])

    for (int i = 0; i < n; ++i) {
      if (a_h[i] != a[i]) {
	printf("Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, a_h[i], a[i]);
	return 1;
      }
    }
  } // loop 'n'
  printf("Succeeded\n");

  // 11. dist_schedule static chunk, schedule static chunk
  printf("dist_schedule static chunk, schedule static chunk\n");
  for (int n = 32 ; n < MAX_N ; n+=5000) {
    int t = 0;
    reset_input(a, a_h, b, c);

#pragma omp target update to(a[:n],b[:n],c[:n])

    for (int tms = 1 ; tms <= 256 ; tms *= 2) { // 8 times
      for (int ths = 32 ; ths <= 1024 ; ths *= 2) { // 6 times
	for(int dssch = 128 ; dssch <= n ; dssch *= 1200) {
	  for(int sch = 100 ; sch <= n ; sch *= 3000) {
	    t++;
#pragma omp target
#pragma omp teams num_teams(tms) thread_limit(ths)
	    {
#pragma omp distribute parallel for dist_schedule(static,dssch) schedule(static,sch)
	      for (int i = 0; i < n; ++i) {
		a[i] += b[i] + c[i];
	      }
	    }
	  } // loop 'sch'
	} // loop 'dssch'
      } // loop 'ths'
    } // loop 'tms'

    // check results for each 'n'
    for (int times = 0 ; times < t ; times++)
      for (int i = 0; i < n; ++i)
	a_h[i] += b[i] + c[i];

#pragma omp target update from(a[:n])

    for (int i = 0; i < n; ++i) {
      if (a_h[i] != a[i]) {
	printf("Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, a_h[i], a[i]);
	return 1;
      }
    }
  } // loop 'n'
  printf("Succeeded\n");

  // 12. dist_schedule static chunk, schedule dynamic no chunk
  printf("dist_schedule static chunk, schedule dynamic no chunk\n");
  for (int n = 32 ; n < MAX_N ; n+=5000) {
    int t = 0;
    reset_input(a, a_h, b, c);

#pragma omp target update to(a[:n],b[:n],c[:n])

    for (int tms = 1 ; tms <= 256 ; tms *= 2) { // 8 times
      for (int ths = 32 ; ths <= 1024 ; ths *= 2) { // 6 times
	for(int sch = 128 ; sch <= n ; sch *= 3000) {
	  t++;
#pragma omp target
#pragma omp teams num_teams(tms) thread_limit(ths)
	  {
#pragma omp distribute parallel for dist_schedule(static,sch) schedule(dynamic)
	    for (int i = 0; i < n; ++i) {
	      a[i] += b[i] + c[i];
	    }
	  }
	} // loop 'sch'
      } // loop 'ths'
    } // loop 'tms'

    // check results for each 'n'
    for (int times = 0 ; times < t ; times++)
      for (int i = 0; i < n; ++i)
	a_h[i] += b[i] + c[i];

#pragma omp target update from(a[:n])

    for (int i = 0; i < n; ++i) {
      if (a_h[i] != a[i]) {
	printf("Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, a_h[i], a[i]);
	return 1;
      }
    }
  } // loop 'n'
  printf("Succeeded\n");

  // 13. dist_schedule static chunk, schedule dynamic chunk
  printf("dist_schedule static chunk, schedule dynamic chunk\n");
  for (int n = 32 ; n < MAX_N ; n+=5000) {
    int t = 0;
    reset_input(a, a_h, b, c);

#pragma omp target update to(a[:n],b[:n],c[:n])

    for (int tms = 1 ; tms <= 256 ; tms *= 2) { // 8 times
      for (int ths = 32 ; ths <= 1024 ; ths *= 2) { // 6 times
	for(int dssch = 128 ; dssch <= n ; dssch *= 3000) {
	  for(int sch = 1000 ; sch <= n ; sch *= 3000) {
	    t++;
#pragma omp target
#pragma omp teams num_teams(tms) thread_limit(ths)
	    {
#pragma omp distribute parallel for dist_schedule(static,dssch) schedule(dynamic,sch)
	      for (int i = 0; i < n; ++i) {
		a[i] += b[i] + c[i];
	      }
	    }
	  } // loop 'sch'
	} // loop 'dssch'
      } // loop 'ths'
    } // loop 'tms'

    // check results for each 'n'
    for (int times = 0 ; times < t ; times++)
      for (int i = 0; i < n; ++i)
	a_h[i] += b[i] + c[i];

#pragma omp target update from(a[:n])

    for (int i = 0; i < n; ++i) {
      if (a_h[i] != a[i]) {
	printf("Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, a_h[i], a[i]);
	return 1;
      }
    }
  } // loop 'n'
  printf("Succeeded\n");

  // 14. dist_schedule static chunk, schedule guided no chunk
  printf("dist_schedule static chunk, schedule guided no chunk\n");
  for (int n = 32 ; n < MAX_N ; n+=5000) {
    int t = 0;
    reset_input(a, a_h, b, c);

#pragma omp target update to(a[:n],b[:n],c[:n])

    for (int tms = 1 ; tms <= 256 ; tms *= 2) { // 8 times
      for (int ths = 32 ; ths <= 1024 ; ths *= 2) { // 6 times
	for(int sch = 1000 ; sch <= n ; sch *= 3000) {
	  t++;
#pragma omp target
#pragma omp teams num_teams(tms) thread_limit(ths)
	  {
#pragma omp distribute parallel for dist_schedule(static,sch) schedule(guided)
	    for (int i = 0; i < n; ++i) {
	      a[i] += b[i] + c[i];
	    }
	  }
	} // loop 'sch'
      } // loop 'ths'
    } // loop 'tms'

    // check results for each 'n'
    for (int times = 0 ; times < t ; times++)
      for (int i = 0; i < n; ++i)
	a_h[i] += b[i] + c[i];

#pragma omp target update from(a[:n])

    for (int i = 0; i < n; ++i) {
      if (a_h[i] != a[i]) {
	printf("Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, a_h[i], a[i]);
	return 1;
      }
    }
  } // loop 'n'
  printf("Succeeded\n");

  // 15. dist_schedule static chunk, schedule guided chunk
  printf("dist_schedule static chunk, schedule guided chunk\n");
  for (int n = 32 ; n < MAX_N ; n+=5000) {
    int t = 0;
    reset_input(a, a_h, b, c);

#pragma omp target update to(a[:n],b[:n],c[:n])

    for (int tms = 1 ; tms <= 256 ; tms *= 2) { // 8 times
      for (int ths = 32 ; ths <= 1024 ; ths *= 2) { // 6 times
	for(int dssch = 1000 ; dssch <= n ; dssch *= 3000) {
	  for(int sch = 1000 ; sch <= n ; sch *= 3000) {
	    t++;
#pragma omp target
#pragma omp teams num_teams(tms) thread_limit(ths)
	    {
#pragma omp distribute parallel for dist_schedule(static,dssch) schedule(guided,sch)
	      for (int i = 0; i < n; ++i) {
		a[i] += b[i] + c[i];
	      }
	    }
	  } // loop 'sch'
	} // loop 'dssch'
      } // loop 'ths'
    } // loop 'tms'

    // check results for each 'n'
    for (int times = 0 ; times < t ; times++)
      for (int i = 0; i < n; ++i)
	a_h[i] += b[i] + c[i];

#pragma omp target update from(a[:n])

    for (int i = 0; i < n; ++i) {
      if (a_h[i] != a[i]) {
	printf("Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, a_h[i], a[i]);
	return 1;
      }
    }
  } // loop 'n'
  printf("Succeeded\n");


  // 16. dist_schedule static chunk, schedule auto
  printf("dist_schedule static chunk, schedule auto\n");
  for (int n = 32 ; n < MAX_N ; n+=5000) {
    int t = 0;
    reset_input(a, a_h, b, c);

#pragma omp target update to(a[:n],b[:n],c[:n])

    for (int tms = 1 ; tms <= 256 ; tms *= 2) { // 8 times
      for (int ths = 32 ; ths <= 1024 ; ths *= 2) { // 6 times
	for(int sch = 1000 ; sch <= n ; sch *= 3000) {
	  t++;
#pragma omp target
#pragma omp teams num_teams(tms) thread_limit(ths)
	  {
#pragma omp distribute parallel for dist_schedule(static,sch) schedule(auto)
	    for (int i = 0; i < n; ++i) {
	      a[i] += b[i] + c[i];
	    }
	  }
	} // loop 'sch'
      } // loop 'ths'
    } // loop 'tms'

    // check results for each 'n'
    for (int times = 0 ; times < t ; times++)
      for (int i = 0; i < n; ++i)
	a_h[i] += b[i] + c[i];

#pragma omp target update from(a[:n])

    for (int i = 0; i < n; ++i) {
      if (a_h[i] != a[i]) {
	printf("Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, a_h[i], a[i]);
	return 1;
      }
    }
  } // loop 'n'
  printf("Succeeded\n");

    // 17. dist_schedule static chunk, schedule runtime
  printf("dist_schedule static chunk, schedule runtime\n");
  for (int n = 32 ; n < MAX_N ; n+=5000) {
    int t = 0;
    reset_input(a, a_h, b, c);

#pragma omp target update to(a[:n],b[:n],c[:n])

    for (int tms = 1 ; tms <= 256 ; tms *= 2) { // 8 times
      for (int ths = 32 ; ths <= 1024 ; ths *= 2) { // 6 times
	for(int sch = 1000 ; sch <= n ; sch *= 3000) {
	  t++;
#pragma omp target
#pragma omp teams num_teams(tms) thread_limit(ths)
	  {
#pragma omp distribute parallel for dist_schedule(static,sch) schedule(runtime)
	    for (int i = 0; i < n; ++i) {
	      a[i] += b[i] + c[i];
	    }
	  }
	} // loop 'sch'
      } // loop 'ths'
    } // loop 'tms'

    // check results for each 'n'
    for (int times = 0 ; times < t ; times++)
      for (int i = 0; i < n; ++i)
	a_h[i] += b[i] + c[i];

#pragma omp target update from(a[:n])

    for (int i = 0; i < n; ++i) {
      if (a_h[i] != a[i]) {
	printf("Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, a_h[i], a[i]);
	return 1;
      }
    }
  } // loop 'n'
  printf("Succeeded\n");

#pragma omp target exit data map(release:a[:MAX_N],b[:MAX_N],c[:MAX_N])

  return 0;
}
