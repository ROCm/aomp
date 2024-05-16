#if defined(__cplusplus)
extern "C" {
#endif

#pragma omp declare target
void inc_arrayval(int i, int *array) ;
void dec_arrayval(int i, int *array) ; 
#pragma omp end declare target

#if defined(__cplusplus)
}
#endif
