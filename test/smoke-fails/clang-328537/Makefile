include ../../Makefile.defs

TESTNAME     = clang-328537
TESTSRC_MAIN = clang-328537.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)

HSA_XNACK    ?= 1
SUPPORTED    = $(SUPPORTS_USM)

#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules

run:
	HSA_XNACK=${HSA_XNACK} ./$(TESTNAME)
	@echo "Test will fail on below MI200"
