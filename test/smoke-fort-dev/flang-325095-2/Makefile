NOOPT        = 1
include ../../Makefile.defs

TESTNAME     = use_dev_addr_fort
TESTSRC_MAIN = use_dev_addr_fort.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
CFLAGS       =
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)

include ../Makefile.rules
