#include <stdio.h>
extern void __kmpc_rfun_sum_lds_i(int *val, int *otherval) { *val += *otherval; } 
extern void __kmpc_rfun_sum_i (int *val, int otherval) { *val += otherval; }
extern void __kmpc_xteamr_i_4x64_fast_sum(int v, int *r_p, int *tvs, unsigned int *td,
                     void (*rf)(int *, int),
                     void (*rflds)(int *, int *), const int rnv,
                     const unsigned long int k, const unsigned int nt) {
fprintf(stderr,"\n\n\n  WARNING  WITH MANDATORY __kmpc_xteamr_i_4x64_fast_sum should never be called by CPU\n\n");
}
extern void __kmpc_xteamr_i_4x64(int v, int *r_p, int *tvs, unsigned int *td,
                     void (*rf)(int *, int),
                     void (*rflds)(int *, int *), const int rnv,
                     const unsigned long int k, const unsigned int nt) {
fprintf(stderr,"\n\n\n  WARNING  WITH MANDATORY __kmpc_xteamr_i_4x64 should never be called by CPU\n\n");
//  _xteam_reduction<int, 4, 64>(v, r_p, tvs, td, rf, rflds, rnv, k, nt);
}
