include ../../Makefile.defs

TESTNAME     = prepinned-ompt-target-tracing
TESTSRC_MAIN = memory_mgmt_pinned_predefined.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

RUNENV       += OMPTEST_USE_OMPT_EMI=1
RUNENV       += OMPTEST_USE_OMPT_TRACING=1
RUNCMD       = ./$(TESTNAME) | $(FILECHECK) $(TESTSRC_MAIN)

CLANG        = clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE) $(OMPTEST)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
