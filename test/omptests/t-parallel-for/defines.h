
#undef PARALLEL_FOR
#define PARALLEL_FOR(PRE,X,POST,VERIFY) TEST({ \
PRE  \
_Pragma("omp parallel for if(threads[0] > 1) num_threads(threads[0]) PARALLEL_FOR_CLAUSES") \
  X  \
_Pragma("omp parallel for schedule(auto) if(threads[0] > 1) num_threads(threads[0]) PARALLEL_FOR_CLAUSES") \
  X  \
_Pragma("omp parallel for schedule(dynamic) if(threads[0] > 1) num_threads(threads[0]) PARALLEL_FOR_CLAUSES") \
  X  \
_Pragma("omp parallel for schedule(guided) if(threads[0] > 1) num_threads(threads[0]) PARALLEL_FOR_CLAUSES") \
  X  \
_Pragma("omp parallel for schedule(runtime) if(threads[0] > 1) num_threads(threads[0]) PARALLEL_FOR_CLAUSES") \
  X  \
_Pragma("omp parallel for schedule(static) if(threads[0] > 1) num_threads(threads[0]) PARALLEL_FOR_CLAUSES") \
  X  \
_Pragma("omp parallel for schedule(static,1) if(threads[0] > 1) num_threads(threads[0]) PARALLEL_FOR_CLAUSES") \
  X  \
_Pragma("omp parallel for schedule(static,9) if(threads[0] > 1) num_threads(threads[0]) PARALLEL_FOR_CLAUSES") \
  X  \
_Pragma("omp parallel for schedule(static,30000) if(threads[0] > 1) num_threads(threads[0]) PARALLEL_FOR_CLAUSES") \
  X  \
POST \
}, VERIFY)

#undef SUMS
#define SUMS (9)

