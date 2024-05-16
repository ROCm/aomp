include ../../Makefile.defs
HSA_XNACK ?= 1
AOMP_TARGET_FEATURES = :xnack+

TESTNAME     = managed_memory
TESTSRC_MAIN = managed_memory.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

SUPPORTED    = $(SUPPORTS_USM)

CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)

RUNENV       += HSA_XNACK=$(HSA_XNACK)
RUNCMD       = ./$(TESTNAME)


#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
