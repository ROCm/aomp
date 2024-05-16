include ../../Makefile.defs

TESTNAME     = flang-367296
TESTSRC_MAIN = flang-367296.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)

SUPPORTED    = $(SUPPORTS_USM)

#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
