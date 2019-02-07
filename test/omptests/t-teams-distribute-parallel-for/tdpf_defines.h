
// Due to compiler limitations, we cannot have more than one target region in a macro.

#undef TDPARALLEL_FOR1
#define TDPARALLEL_FOR1(PRE,X,POST,VERIFY) { \
TESTD2("omp target", \
PRE, { \
_Pragma("omp teams distribute parallel for num_teams(num_teams) if(threads[0] >= 1) num_threads(threads[0]) TDPARALLEL_FOR_CLAUSES") \
X \
}, POST, VERIFY) \
}

#undef TDPARALLEL_FOR2
#define TDPARALLEL_FOR2(PRE,X,POST,VERIFY) { \
TESTD2("omp target teams distribute parallel for schedule(auto) num_teams(num_teams) if(threads[0] >= 1) num_threads(threads[0]) TDPARALLEL_FOR_CLAUSES", \
PRE, X, POST, VERIFY) \
}

#undef TDPARALLEL_FOR3
#define TDPARALLEL_FOR3(PRE,X,POST,VERIFY) { \
TESTD2("omp target teams distribute parallel for schedule(dynamic) num_teams(num_teams) if(threads[0] >= 1) num_threads(threads[0]) TDPARALLEL_FOR_CLAUSES", \
PRE, X, POST, VERIFY) \
}

#undef TDPARALLEL_FOR4
#define TDPARALLEL_FOR4(PRE,X,POST,VERIFY) { \
TESTD2("omp target teams distribute parallel for schedule(guided) num_teams(num_teams) if(threads[0] >= 1) num_threads(threads[0]) TDPARALLEL_FOR_CLAUSES", \
PRE, X, POST, VERIFY) \
}

#undef TDPARALLEL_FOR5
#define TDPARALLEL_FOR5(PRE,X,POST,VERIFY) { \
TESTD2("omp target teams distribute parallel for schedule(runtime) num_teams(num_teams) if(threads[0] >= 1) num_threads(threads[0]) TDPARALLEL_FOR_CLAUSES", \
PRE, X, POST, VERIFY) \
}

#undef TDPARALLEL_FOR6
#define TDPARALLEL_FOR6(PRE,X,POST,VERIFY) { \
TESTD2("omp target teams distribute parallel for schedule(static) num_teams(num_teams) if(threads[0] >= 1) num_threads(threads[0]) TDPARALLEL_FOR_CLAUSES", \
PRE, X, POST, VERIFY) \
}

