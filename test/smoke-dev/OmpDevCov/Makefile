include ../../Makefile.defs

# OmpDevCov: device-side PGO round trip for OpenMP target offload. The harness
# builds omp_pgo (TESTNAME) with -fprofile-generate; run_and_check.sh runs it,
# verifies the device counters were drained via the HSA pass, merges, and feeds
# them back with -fprofile-use.

TESTNAME     = omp_pgo
TESTSRC_MAIN = omp_pgo.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

# Extra budget: run_and_check.sh also performs the -fprofile-use rebuild + run.
TIMEOUT      = 300s

CFLAGS       = -fopenmp $(TARGET) -fprofile-generate
LINK_FLAGS   = -Wl,-rpath,$(AOMP)/lib

CC           = $(AOMP)/bin/clang $(VERBOSE)

RUNCMD       = ./run_and_check.sh "$(AOMP)" "$(GPU_W_FEATURES)" "$(FILECHECK)"

include ../Makefile.rules

clean::
	rm -f omp_pgo omp_pgo_use *.profraw merged.profdata
