include ../../Makefile.defs

TESTNAME     = veccopy-ompt-target-emi-tracing
TESTSRC_MAIN = veccopy-ompt-target-emi-tracing.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

RUNENV       += OMPTEST_USE_OMPT_EMI=1
RUNENV       += OMPTEST_USE_OMPT_TRACING=1
RUNCMD       = ./$(TESTNAME) | $(FILECHECK) $(TESTSRC_MAIN)

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE) $(OMPTEST)

include ../Makefile.rules
