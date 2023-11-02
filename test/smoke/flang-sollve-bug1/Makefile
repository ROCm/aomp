include ../../Makefile.defs

TESTNAME     = flang-sollve-bug1
TESTSRC_MAIN = flang-sollve-bug1.F90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
CFLAGS = -O0

include ../Makefile.rules
