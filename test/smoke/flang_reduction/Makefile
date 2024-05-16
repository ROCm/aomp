include ../../Makefile.defs

TESTNAME     = flang_reduction
TESTSRC_MAIN = flang_reduction.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
OMP_FLAGS   += -fopenmp-target-new-runtime

include ../Makefile.rules
