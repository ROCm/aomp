include ../../Makefile.defs
HSA_XNACK        ?= 1
AOMP_TARGET_FEATURES = :xnack+

TESTNAME     = omp_target_is
TESTSRC_MAIN = omp_target_is.cpp
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
