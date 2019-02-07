
// Due to compiler limitations, we cannot have more than one target region in a macro.

#undef TARGET_TDPARALLEL_FOR1
#define TARGET_TDPARALLEL_FOR1(PRE,X,POST,VERIFY) { \
TESTD2("omp target teams distribute parallel for if(threads[0] >= 1) num_threads(threads[0]) num_teams(num_teams) TARGET_TDPARALLEL_FOR_CLAUSES", \
PRE, X, POST, VERIFY) \
}

#undef TARGET_TDPARALLEL_FOR2
#define TARGET_TDPARALLEL_FOR2(PRE,X,POST,VERIFY) { \
TESTD2("omp target teams distribute parallel for schedule(auto) if(threads[0] >= 1) num_threads(threads[0]) num_teams(num_teams) TARGET_TDPARALLEL_FOR_CLAUSES", \
PRE, X, POST, VERIFY) \
}

#undef TARGET_TDPARALLEL_FOR3
#define TARGET_TDPARALLEL_FOR3(PRE,X,POST,VERIFY) { \
TESTD2("omp target teams distribute parallel for schedule(dynamic) if(threads[0] >= 1) num_threads(threads[0]) num_teams(num_teams) TARGET_TDPARALLEL_FOR_CLAUSES", \
PRE, X, POST, VERIFY) \
}

#undef TARGET_TDPARALLEL_FOR4
#define TARGET_TDPARALLEL_FOR4(PRE,X,POST,VERIFY) { \
TESTD2("omp target teams distribute parallel for schedule(guided) if(threads[0] >= 1) num_threads(threads[0]) num_teams(num_teams) TARGET_TDPARALLEL_FOR_CLAUSES", \
PRE, X, POST, VERIFY) \
}

#undef TARGET_TDPARALLEL_FOR5
#define TARGET_TDPARALLEL_FOR5(PRE,X,POST,VERIFY) { \
TESTD2("omp target teams distribute parallel for schedule(runtime) if(threads[0] >= 1) num_threads(threads[0]) num_teams(num_teams) TARGET_TDPARALLEL_FOR_CLAUSES", \
PRE, X, POST, VERIFY) \
}

#undef TARGET_TDPARALLEL_FOR6
#define TARGET_TDPARALLEL_FOR6(PRE,X,POST,VERIFY) { \
TESTD2("omp target teams distribute parallel for schedule(static) if(threads[0] >= 1) num_threads(threads[0]) num_teams(num_teams) TARGET_TDPARALLEL_FOR_CLAUSES", \
PRE, X, POST, VERIFY) \
}

