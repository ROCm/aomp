include ../../Makefile.defs

TESTNAME     = flang-333423
TESTSRC_MAIN = flang-333423.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        = flang -g
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)

include ../Makefile.rules
