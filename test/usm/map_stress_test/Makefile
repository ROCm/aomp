include ../../Makefile.defs

TESTNAME     = map_stress_test
TESTSRC_MAIN = map_stress_test.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang -offload-usm
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules

run:
	HSA_XNACK=1 ./map_stress_test 2>&1 | tee run.log
