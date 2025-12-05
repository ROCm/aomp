include ../../Makefile.defs

TESTNAME     = clang-392854
TESTSRC_MAIN = clang-392854.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
RPTH = -Wl,-rpath,$(AOMPHIP)/lib

CLANG        = amdclang++
OMP_FLAGS    = -fopenmp -x hip $(RPTH)
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = HIPCC_VERBOSE=1 $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules

