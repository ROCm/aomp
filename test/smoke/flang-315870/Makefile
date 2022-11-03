include ../../Makefile.defs

TESTNAME     = flang-315870
TESTSRC_MAIN = flang-315870.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang -mcode-object-version=5
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
CFLAGS       =
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
