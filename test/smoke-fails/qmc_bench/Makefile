include ../../Makefile.defs

TESTNAME     = qmc_bench
TESTSRC_MAIN = qmc_bench.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules

run: $(TESTNAME)
	export OMP_NUM_THREADS=128
	$(RUNENV) $(RUNPROF) ./$(TESTNAME) 2>&1 | tee $@.log
	unset OMP_NUM_THREADS
