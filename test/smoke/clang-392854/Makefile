include ../../Makefile.defs

TESTNAME     = clang-392854
TESTSRC_MAIN = clang-392854.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = amdclang++
OMP_FLAGS    = -fopenmp -x hip
OMP_BIN      = $(AOMPHIP)/bin/$(CLANG)
CC           = HIPCC_VERBOSE=1 $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules

