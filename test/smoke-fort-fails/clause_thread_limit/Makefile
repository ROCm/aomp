include ../../Makefile.defs

TESTNAME     = atest
TESTSRC_MAIN = atest.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang-new
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#OMP_FLAGS   +=
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
