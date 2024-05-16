include ../../Makefile.defs

TESTNAME     = mixed_maps
TESTSRC_MAIN = mixed_maps.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)

HSA_XNACK ?= 1
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules

run:
	HSA_XNACK=${HSA_XNACK} ./$(TESTNAME)
