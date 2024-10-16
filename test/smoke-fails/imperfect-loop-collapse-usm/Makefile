include ../../Makefile.defs

TESTNAME     = imperfect_loop_collapse_usm
TESTSRC_MAIN = imperfect_loop_collapse_usm.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)

HSA_XNACK ?= 1
SUPPORTED    = $(SUPPORTS_USM)

# Our "run" target gets overridden.  Make sure we run with HSA_XNACK set
# appropriately.
RUNENV	     += HSA_XNACK=${HSA_XNACK}

#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules

run:
	HSA_XNACK=${HSA_XNACK} ./$(TESTNAME)
