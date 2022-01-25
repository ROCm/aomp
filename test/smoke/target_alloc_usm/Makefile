include ../../Makefile.defs

TESTNAME     = target_alloc_usm
TESTSRC_MAIN = target_alloc_usm.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules

run:
	HSA_XNACK=1 ./$(TESTNAME) 2>&1 | tee $@.log
