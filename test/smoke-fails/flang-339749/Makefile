include ../../Makefile.defs

TESTNAME     = flang-339749
TESTSRC_MAIN = flang-339749.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        = flang -DUSE_REDUCE
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)

include ../Makefile.rules
