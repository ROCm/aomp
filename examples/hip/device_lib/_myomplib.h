//
// _mylib.h : Headers for demo library _mylib.  Functions in this library can be used for 
//            either HIP or OpenMP Target regions.
//
#if defined(__cplusplus)
extern "C" {
#endif

__device__ __host__ void inc_omp(int i, int *array) ;
__device__ __host__ void dec_omp(int i, int *array) ;

#if defined(__cplusplus)
}
#endif
