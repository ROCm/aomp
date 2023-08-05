include ../../Makefile.defs

TESTNAME     = managed_memory
TESTSRC_MAIN = managed_memory.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

HSA_XNACK ?= 1

CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
RUNENV       = HSA_XNACK=${HSA_XNACK}
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
