include ../../Makefile.defs

TESTNAME     = target-shared-variables
TESTSRC_MAIN = target-shared-variables.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

EXTRA_OMP_FLAGS += -mllvm -openmp-opt-disable-deglobalization=true
CLANG           = clang++
OMP_BIN         = $(AOMP)/bin/$(CLANG)
CC              = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
