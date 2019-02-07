
#undef PARALLEL_FOR
#define PARALLEL_FOR(PRE,X,POST,VERIFY) TEST({ \
PRE  \
_Pragma("omp parallel for simd if(threads[0] > 1) num_threads(threads[0]) PARALLEL_FOR_CLAUSES") \
  X  \
_Pragma("omp parallel for simd schedule(auto) if(threads[0] > 1) num_threads(threads[0]) PARALLEL_FOR_CLAUSES") \
  X  \
_Pragma("omp parallel for simd schedule(dynamic) if(threads[0] > 1) num_threads(threads[0]) PARALLEL_FOR_CLAUSES") \
  X  \
_Pragma("omp parallel for simd schedule(guided) if(threads[0] > 1) num_threads(threads[0]) PARALLEL_FOR_CLAUSES") \
  X  \
_Pragma("omp parallel for simd schedule(runtime) if(threads[0] > 1) num_threads(threads[0]) PARALLEL_FOR_CLAUSES") \
  X  \
_Pragma("omp parallel for simd schedule(static) if(threads[0] > 1) num_threads(threads[0]) PARALLEL_FOR_CLAUSES") \
  X  \
_Pragma("omp parallel for simd schedule(static,1) if(threads[0] > 1) num_threads(threads[0]) PARALLEL_FOR_CLAUSES") \
  X  \
_Pragma("omp parallel for simd schedule(static,9) if(threads[0] > 1) num_threads(threads[0]) PARALLEL_FOR_CLAUSES") \
  X  \
_Pragma("omp parallel for simd schedule(static,30000) if(threads[0] > 1) num_threads(threads[0]) PARALLEL_FOR_CLAUSES") \
  X  \
POST \
}, VERIFY)

#undef SUMS
#define SUMS (9)

