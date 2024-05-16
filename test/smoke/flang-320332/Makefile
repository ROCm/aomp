include ../../Makefile.defs

TESTNAME     = flang-320332
TESTSRC_MAIN = flang-320332.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)

include ../Makefile.rules
