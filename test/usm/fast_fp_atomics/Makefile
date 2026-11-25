include ../../Makefile.defs

TESTNAME     = fast_fp_atomics
TESTSRC_MAIN = fast_fp_atomics.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang++ -offload-usm
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)

include ../Makefile.rules

run:
	HSA_XNACK=1 ./fast_fp_atomics 2>&1 | tee run.log
