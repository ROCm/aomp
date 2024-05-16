include ../../Makefile.defs

TESTNAME     = target_alloc
TESTSRC_MAIN = target_alloc.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

HSA_XNACK ?= 1

include ../Makefile.rules

run:
	HSA_XNACK=${HSA_XNACK} ./$(TESTNAME)
