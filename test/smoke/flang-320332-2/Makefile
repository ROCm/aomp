include ../../Makefile.defs

TESTNAME     = flang-320332-2
TESTSRC_MAIN = flang-320332-2.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)

OMP_FLAGS   += -DLEN_2 -DENTER_DATA

include ../Makefile.rules
