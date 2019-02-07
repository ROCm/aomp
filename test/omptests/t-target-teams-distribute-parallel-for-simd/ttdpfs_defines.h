
// Due to compiler limitations, we cannot have more than one target region in a macro.

#undef TARGET_TDPARALLEL_FOR_SIMD1
#define TARGET_TDPARALLEL_FOR_SIMD1(PRE,X,POST,VERIFY) { \
TESTD2("omp target teams distribute parallel for simd if(threads[0] >= 1) num_threads(threads[0]) num_teams(num_teams) TARGET_TDPARALLEL_FOR_SIMD_CLAUSES", \
PRE, X, POST, VERIFY) \
}

#undef TARGET_TDPARALLEL_FOR_SIMD2
#define TARGET_TDPARALLEL_FOR_SIMD2(PRE,X,POST,VERIFY) { \
TESTD2("omp target teams distribute parallel for simd schedule(auto) if(threads[0] >= 1) num_threads(threads[0]) num_teams(num_teams) TARGET_TDPARALLEL_FOR_SIMD_CLAUSES", \
PRE, X, POST, VERIFY) \
}

#undef TARGET_TDPARALLEL_FOR_SIMD3
#define TARGET_TDPARALLEL_FOR_SIMD3(PRE,X,POST,VERIFY) { \
TESTD2("omp target teams distribute parallel for simd schedule(dynamic) if(threads[0] >= 1) num_threads(threads[0]) num_teams(num_teams) TARGET_TDPARALLEL_FOR_SIMD_CLAUSES", \
PRE, X, POST, VERIFY) \
}

#undef TARGET_TDPARALLEL_FOR_SIMD4
#define TARGET_TDPARALLEL_FOR_SIMD4(PRE,X,POST,VERIFY) { \
TESTD2("omp target teams distribute parallel for simd schedule(guided) if(threads[0] >= 1) num_threads(threads[0]) num_teams(num_teams) TARGET_TDPARALLEL_FOR_SIMD_CLAUSES", \
PRE, X, POST, VERIFY) \
}

#undef TARGET_TDPARALLEL_FOR_SIMD5
#define TARGET_TDPARALLEL_FOR_SIMD5(PRE,X,POST,VERIFY) { \
TESTD2("omp target teams distribute parallel for simd schedule(runtime) if(threads[0] >= 1) num_threads(threads[0]) num_teams(num_teams) TARGET_TDPARALLEL_FOR_SIMD_CLAUSES", \
PRE, X, POST, VERIFY) \
}

#undef TARGET_TDPARALLEL_FOR_SIMD6
#define TARGET_TDPARALLEL_FOR_SIMD6(PRE,X,POST,VERIFY) { \
TESTD2("omp target teams distribute parallel for simd schedule(static) if(threads[0] >= 1) num_threads(threads[0]) num_teams(num_teams) TARGET_TDPARALLEL_FOR_SIMD_CLAUSES", \
PRE, X, POST, VERIFY) \
}

