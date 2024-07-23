include ../../Makefile.defs

TESTNAME     = target_alloc_fast_fp_atomics
TESTSRC_MAIN = target_alloc_fast_fp_atomics.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
CFLAGS       = -O3
#-ccc-print-phases
#"-\#\#\#"

HSA_XNACK ?= 1

include ../Makefile.rules

run:
	HSA_XNACK=${HSA_XNACK} ./$(TESTNAME)
