#include <stdio.h>
#include <omp.h>

extern void __kmpc_rfun_sum_lds_i(int *val, int *otherval);
extern void __kmpc_rfun_sum_i (int *val, int otherval);
extern void __kmpc_xteamr_i_4x64(int v, int *r_p, int *tvs, unsigned int *td,
                     void (*rf)(int *, int),
                     void (*rflds)(int *, int *), const int rnv,
                     const unsigned long int k, const unsigned int nt) ;
extern void __kmpc_xteamr_i_4x64_fast_sum(int v, int *r_p, int *tvs, 
		     unsigned int *td, void (*rf)(int *, int),
                     void (*rflds)(int *, int *), const int rnv,
                     const unsigned long int k, const unsigned int nt) ;
int main() {
  int devid = 0;
  // int nteams = 60;
  int nteams = 1;
  int nthreads = 256;
  int result_i = 0;
  int tvals[60];
  //int sz = 19200;
  int sz = 5;
  int validate_i = ((sz + 1) * sz) / 2; // Sum of integers from 1 to sz
  unsigned int teams_done = 0;
  // printf(" tvals0 tvals1 tvals59= %d %d %d\n",tvals[0], tvals[1], tvals[59]);

#pragma omp target teams distribute parallel for num_teams(nteams)   \
  num_threads(nthreads) map(tofrom :result_i) map(from: tvals) device(devid)
  for (unsigned long int k = 0; k < (nteams * nthreads); k++) {
    int val0_i = 0; 
    for (unsigned long int i = k; i < sz; i += (nteams*nthreads)) {
      if (k<sz)
        val0_i += i + 1;
    }
    __kmpc_xteamr_i_4x64_fast_sum(val0_i, &result_i, tvals, &teams_done, 
      __kmpc_rfun_sum_i, __kmpc_rfun_sum_lds_i, 0 , k, nteams);
  }

  // when used without fast_sum, intermediate sums are ok but result is not
  // This could be problem with no initialization of lds.
  int check_sum=0;
  for (unsigned int i =0; i<nteams; i++)
    check_sum += tvals[i];
  // printf(" check_sum %d tvals0 tvals1 tvals59= %d %d %d%d\n",
  //   check_sum, tvals[0], tvals[1], tvals[59]);

  printf(" Result is %d\n",result_i);
  printf(" Validate is %d\n",validate_i);
  if ( validate_i == result_i)
    return 2;
  else
    return 0;
}
