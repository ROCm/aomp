
// Due to compiler limitations, we cannot have more than one target region in a macro.

#undef TDPARALLEL_FOR_SIMD1
#define TDPARALLEL_FOR_SIMD1(PRE,X,POST,VERIFY) { \
TESTD2("omp target", \
PRE, { \
_Pragma("omp teams distribute parallel for simd num_teams(num_teams) if(threads[0] >= 1) num_threads(threads[0]) TDPARALLEL_FOR_SIMD_CLAUSES") \
X \
}, POST, VERIFY) \
}

#undef TDPARALLEL_FOR_SIMD2
#define TDPARALLEL_FOR_SIMD2(PRE,X,POST,VERIFY) { \
TESTD2("omp target teams distribute parallel for simd schedule(auto) num_teams(num_teams) if(threads[0] >= 1) num_threads(threads[0]) TDPARALLEL_FOR_SIMD_CLAUSES", \
PRE, X, POST, VERIFY) \
}

#undef TDPARALLEL_FOR_SIMD3
#define TDPARALLEL_FOR_SIMD3(PRE,X,POST,VERIFY) { \
TESTD2("omp target teams distribute parallel for simd schedule(dynamic) num_teams(num_teams) if(threads[0] >= 1) num_threads(threads[0]) TDPARALLEL_FOR_SIMD_CLAUSES", \
PRE, X, POST, VERIFY) \
}

#undef TDPARALLEL_FOR_SIMD4
#define TDPARALLEL_FOR_SIMD4(PRE,X,POST,VERIFY) { \
TESTD2("omp target teams distribute parallel for simd schedule(guided) num_teams(num_teams) if(threads[0] >= 1) num_threads(threads[0]) TDPARALLEL_FOR_SIMD_CLAUSES", \
PRE, X, POST, VERIFY) \
}

#undef TDPARALLEL_FOR_SIMD5
#define TDPARALLEL_FOR_SIMD5(PRE,X,POST,VERIFY) { \
TESTD2("omp target teams distribute parallel for simd schedule(runtime) num_teams(num_teams) if(threads[0] >= 1) num_threads(threads[0]) TDPARALLEL_FOR_SIMD_CLAUSES", \
PRE, X, POST, VERIFY) \
}

#undef TDPARALLEL_FOR_SIMD6
#define TDPARALLEL_FOR_SIMD6(PRE,X,POST,VERIFY) { \
TESTD2("omp target teams distribute parallel for simd schedule(static) num_teams(num_teams) if(threads[0] >= 1) num_threads(threads[0]) TDPARALLEL_FOR_SIMD_CLAUSES", \
PRE, X, POST, VERIFY) \
}

