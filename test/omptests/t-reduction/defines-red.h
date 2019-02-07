
#undef PARALLEL_FOR
#define PARALLEL_FOR(PRE,X,POST,VERIFY) TEST({ \
PRE  \
_Pragma("omp parallel for if(threads > 1) num_threads(threads) CLAUSES") \
  X  \
POST \
PRE  \
_Pragma("omp parallel for schedule(auto) if(threads > 1) num_threads(threads) CLAUSES") \
  X  \
POST \
PRE  \
_Pragma("omp parallel for schedule(dynamic) if(threads > 1) num_threads(threads) CLAUSES") \
  X  \
POST \
PRE  \
_Pragma("omp parallel for schedule(guided) if(threads > 1) num_threads(threads) CLAUSES") \
  X  \
POST \
PRE  \
_Pragma("omp parallel for schedule(runtime) if(threads > 1) num_threads(threads) CLAUSES") \
  X  \
POST \
PRE  \
_Pragma("omp parallel for schedule(static) if(threads > 1) num_threads(threads) CLAUSES") \
  X  \
POST \
PRE  \
_Pragma("omp parallel for schedule(static,9) if(threads > 1) num_threads(threads) CLAUSES") \
  X  \
POST \
}, VERIFY)

#undef PARALLEL_NESTED_FOR
#define PARALLEL_NESTED_FOR(PRE,X,POST,VERIFY) TEST({ \
_Pragma("omp parallel if(threads > 1) num_threads(threads)") \
{ \
PRE  \
_Pragma("omp for CLAUSES") \
  X  \
POST \
PRE  \
_Pragma("omp for schedule(auto) CLAUSES") \
  X  \
POST \
PRE  \
_Pragma("omp for schedule(dynamic) CLAUSES") \
  X  \
POST \
PRE  \
_Pragma("omp for schedule(guided) CLAUSES") \
  X  \
POST \
PRE  \
_Pragma("omp for schedule(runtime) CLAUSES") \
  X  \
POST \
PRE  \
_Pragma("omp for schedule(static) CLAUSES") \
  X  \
POST \
PRE  \
_Pragma("omp for schedule(static,9) CLAUSES") \
  X  \
POST \
} \
}, VERIFY)

#undef SUMS
#define SUMS (7)

