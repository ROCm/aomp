
#undef NESTED_PARALLEL_FOR
#define NESTED_PARALLEL_FOR(PRE,X,POST,VERIFY) TESTD("omp target parallel num_threads(t)", { \
PRE  \
_Pragma("omp parallel for if(threads[0] > 1) num_threads(threads[0]) NESTED_PARALLEL_FOR_CLAUSES") \
  X  \
_Pragma("omp parallel for schedule(auto) if(threads[0] > 1) num_threads(threads[0]) NESTED_PARALLEL_FOR_CLAUSES") \
  X  \
_Pragma("omp parallel for schedule(static) if(threads[0] > 1) num_threads(threads[0]) NESTED_PARALLEL_FOR_CLAUSES") \
  X  \
_Pragma("omp parallel for schedule(static,9) if(threads[0] > 1) num_threads(threads[0]) NESTED_PARALLEL_FOR_CLAUSES") \
  X  \
POST \
}, VERIFY)

#undef SUMS
#define SUMS (4)

