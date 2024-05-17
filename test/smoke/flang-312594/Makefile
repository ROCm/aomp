include ../../Makefile.defs

TESTNAME     = flang-312594
TESTSRC_MAIN = flang-312594.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE) -DINT_PTR
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
