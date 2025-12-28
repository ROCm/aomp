NOOPT        = 1
NOOMP        = 1
OMP_FLAGS    = -fopenmp
include ../../Makefile.defs

TESTNAME     = crayptr_dsa
TESTSRC_MAIN = crayptr_dsa.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)

include ../Makefile.rules
