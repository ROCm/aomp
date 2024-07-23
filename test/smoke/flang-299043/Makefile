include ../../Makefile.defs

TESTNAME     = flang-299043
TESTSRC_MAIN = flang-299043.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)

UNSUPPORTED = ASAN_RUNTIME

include ../Makefile.rules
