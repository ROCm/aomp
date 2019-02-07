
// Due to compiler limitations, we cannot have more than one target region in a macro.

#undef TARGET_PARALLEL_FOR1
#define TARGET_PARALLEL_FOR1(PRE,X,POST,VERIFY) { \
TESTD2("omp target parallel for simd if(threads[0] > 1) num_threads(threads[0]) TARGET_PARALLEL_FOR_CLAUSES", \
PRE, X, POST, VERIFY) \
}

#undef TARGET_PARALLEL_FOR2
#define TARGET_PARALLEL_FOR2(PRE,X,POST,VERIFY) { \
TESTD2("omp target parallel for simd schedule(auto) if(threads[0] > 1) num_threads(threads[0]) TARGET_PARALLEL_FOR_CLAUSES", \
PRE, X, POST, VERIFY) \
}

#undef TARGET_PARALLEL_FOR3
#define TARGET_PARALLEL_FOR3(PRE,X,POST,VERIFY) { \
TESTD2("omp target parallel for simd schedule(dynamic) if(threads[0] > 1) num_threads(threads[0]) TARGET_PARALLEL_FOR_CLAUSES", \
PRE, X, POST, VERIFY) \
}

#undef TARGET_PARALLEL_FOR4
#define TARGET_PARALLEL_FOR4(PRE,X,POST,VERIFY) { \
TESTD2("omp target parallel for simd schedule(guided) if(threads[0] > 1) num_threads(threads[0]) TARGET_PARALLEL_FOR_CLAUSES", \
PRE, X, POST, VERIFY) \
}

#undef TARGET_PARALLEL_FOR5
#define TARGET_PARALLEL_FOR5(PRE,X,POST,VERIFY) { \
TESTD2("omp target parallel for simd schedule(runtime) if(threads[0] > 1) num_threads(threads[0]) TARGET_PARALLEL_FOR_CLAUSES", \
PRE, X, POST, VERIFY) \
}

#undef TARGET_PARALLEL_FOR6
#define TARGET_PARALLEL_FOR6(PRE,X,POST,VERIFY) { \
TESTD2("omp target parallel for simd schedule(static) if(threads[0] > 1) num_threads(threads[0]) TARGET_PARALLEL_FOR_CLAUSES", \
PRE, X, POST, VERIFY) \
}

#undef TARGET_PARALLEL_FOR7
#define TARGET_PARALLEL_FOR7(PRE,X,POST,VERIFY) { \
TESTD2("omp target parallel for simd schedule(static,1) if(threads[0] > 1) num_threads(threads[0]) TARGET_PARALLEL_FOR_CLAUSES", \
PRE, X, POST, VERIFY) \
}

#undef TARGET_PARALLEL_FOR8
#define TARGET_PARALLEL_FOR8(PRE,X,POST,VERIFY) { \
TESTD2("omp target parallel for simd schedule(static,9) if(threads[0] > 1) num_threads(threads[0]) TARGET_PARALLEL_FOR_CLAUSES", \
PRE, X, POST, VERIFY) \
}

#undef TARGET_PARALLEL_FOR9
#define TARGET_PARALLEL_FOR9(PRE,X,POST,VERIFY) { \
TESTD2("omp target parallel for simd schedule(static,40000) if(threads[0] > 1) num_threads(threads[0]) TARGET_PARALLEL_FOR_CLAUSES", \
PRE, X, POST, VERIFY) \
}

