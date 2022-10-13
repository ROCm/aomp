include ../../Makefile.defs

TESTNAME     = fast_fp_atomics_ompx
TESTSRC_MAIN = fast_fp_atomics_ompx.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

EXTRA_OMP_FLAGS += -fopenmp-version=51
CLANG           ?= clang++
OMP_BIN          = $(AOMP)/bin/$(CLANG)
CC               = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
