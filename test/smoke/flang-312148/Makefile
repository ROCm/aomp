include ../../Makefile.defs

TESTNAME     = flang-312148
TESTSRC_MAIN = flang-312148.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE) -g
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
