include ../../Makefile.defs

TESTNAME     = flang-314146
TESTSRC_MAIN = flang-314146.f
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
OMP_FLAGS   += -g

include ../Makefile.rules
