include ../../Makefile.defs

TESTNAME     = flang-339906
TESTSRC_MAIN = main.f90
TESTSRC_AUX  = matrix.f90
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        = flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)

include ../Makefile.rules
