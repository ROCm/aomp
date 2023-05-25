include ../../Makefile.defs

TESTNAME     = managed_memory
TESTSRC_MAIN = managed_memory.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
RUNENV       = HSA_XNACK=1
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
