include ../../Makefile.defs

TESTNAME     = tgt-generic-parallel-reduction
TESTSRC_MAIN = tgt-generic-parallel-reduction.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)

include ../Makefile.rules
