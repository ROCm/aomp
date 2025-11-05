NOOPT        = 1
include ../../Makefile.defs

TESTNAME     = tgt-variable-array-dimension
TESTSRC_MAIN = tgt-variable-array-dimension.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
CFLAGS       = -O0

include ../Makefile.rules
