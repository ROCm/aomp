include ../../Makefile.defs

TESTNAME     = clang-meta
TESTSRC_MAIN = clang-meta.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang++
CFLAGS       = -DSPEC_TARGET_OPENMP -fopenmp-version=52
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
