#include <stdio.h>
#include <omp.h>

#define N 10000000 //N must be even

void init(long long n, double *v1, double *v2) {
  for(long long i = 0; i < n; i++) {
    v1[i] = i;
    v2[i] = i*2;
  }
}

long long check(long long n, double *vxv, double *v1, double *v2) {
  long long err = 0;
  for(long long i = 0; i < n; i++)
    if (vxv[i] != v1[i]*v2[i]) {
      err++;
      printf("vxv[%lld] = %lf, should be %lf\n", i, vxv[i], v1[i]*v2[i]);
      if (err > 10) break;
    }

  return err;
}
int main(){
  long long n=N;
  double K = 29294874.2938;
  double L = 223456774.6544;
  long long chunk=1000;
  double *v1 = new double[N];
  double *v2 = new double[N];
  double *vxv = new double[N];

  init(n, v1,v2);

  long long n1 = n-10;
  double *z = new double[n]; // just to create some extra work
  double scalar = -1.0;
  int scalar_err = 0;
  
  printf("Now going into parallel\n");
  fflush(stdout);
  #pragma omp parallel num_threads(10)
  {
    //    #pragma omp masked // teams distribute
    if (omp_get_thread_num() == 0) {
      printf("Thread 0 is going to launch a kernel\n");
      fflush(stdout);
      // nowait means: create a target task
      // Thread 0 will be stuck waiting for the target region to complete
      // it does not mean it continues executing
      #pragma omp target nowait map(tofrom:z[:n1], scalar) map(to: v1[0:n1]) map(to: v2[0:n1]) map(from: vxv[0:n1])
      for(long long i = 0; i < n1; i++) {
	z[i] = v1[i]*v2[i];
	z[i] *= -1.0;
	z[i] += v1[i]+v2[i]+2.0+v1[i]*K+v2[i]*L;
	z[i] += v2[i]+v2[i]+2.0+v2[i]*K+v2[i]*L;
	z[i] += v1[i]+v2[i]+2.0+v2[i]*K+v2[i]*L;
	z[i] += v1[i]+v1[i]+2.0+v1[i]*K+v2[i]*L/v1[i];
	z[i] += v1[i]+v2[i]+2.0+v1[i]*K+v2[i]*L/v2[i];
	vxv[i] = v1[i]*v2[i];

	// only change scalar value at the very last iteration, hoping it will happen later than the check on the host
	if (i == n1-1)
	  scalar = 13.0;
      }
    }

    printf("Thread %d continuing into for loop\n", omp_get_thread_num());
    fflush(stdout);
    
    // on host
    // - nowait is necessary to prevent an implicit barrier with the target region
    // - static schedule is necessary to ensure thread 0 has to execute some iterations of the for loop
    //   and it does so before the target region above has completed execution
    #pragma omp for nowait schedule(static)
    for(long long i = n1; i < n; i++) {
      printf("Tid = %d: doing iteration %lld\n", omp_get_thread_num(), i);
      vxv[i] = v1[i]*v2[i];
    }

    // ideally, scalar is not visible by thread 1 until the implicit barrier at the end of the parallel
    // region, due to weak memory consistency. However, this depends on how asynchronous execution
    // is actually implemented
    if (omp_get_thread_num() == 1 && scalar != -1.0) {
      scalar_err = 1;
      printf("scalar was modified by target region: might indicate nowait isn't working properly\n");      
      fflush(stdout);      
    }
    printf("Thread %d done with host for loop\n", omp_get_thread_num());
    fflush(stdout);
  } // implicit barrier will wait for target region to complete

  return scalar_err || check(n, vxv, v1, v2);
}
