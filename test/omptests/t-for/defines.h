
#undef PARALLEL
#define PARALLEL(PRE,X,POST,VERIFY) TEST({ \
PRE  \
_Pragma("omp parallel if(threads[0] > 1) num_threads(threads[0])") \
{ \
_Pragma("omp for FOR_CLAUSES") \
  X  \
_Pragma("omp for schedule(auto) FOR_CLAUSES") \
  X  \
_Pragma("omp for schedule(dynamic) FOR_CLAUSES") \
  X  \
_Pragma("omp for schedule(guided) FOR_CLAUSES") \
  X  \
_Pragma("omp for schedule(runtime) FOR_CLAUSES") \
  X  \
_Pragma("omp for schedule(static) FOR_CLAUSES") \
  X  \
_Pragma("omp for schedule(static,1) FOR_CLAUSES") \
  X  \
_Pragma("omp for schedule(static,9) FOR_CLAUSES") \
  X  \
_Pragma("omp for schedule(static,30000) FOR_CLAUSES") \
  X  \
} \
POST \
}, VERIFY)

#undef SUMS
#define SUMS (9)

