include ../../Makefile.defs

TESTNAME     = distribute-parallel-do-if-host
TESTSRC_MAIN = distribute-parallel-do-if-host.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang-new
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
OMP_FLAGS   += -fopenmp-version=52

include ../Makefile.rules
