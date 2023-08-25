include ../../Makefile.defs

TESTNAME     = launch_latency
TESTSRC_MAIN = launch_latency.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
run: $(TESTNAME)
	ulimit -s unlimited ; \
	$(AOMP)/bin/rocprof --stats ./$(TESTNAME) > run.log 2>&1 ; \
	python3 printLatency.py
