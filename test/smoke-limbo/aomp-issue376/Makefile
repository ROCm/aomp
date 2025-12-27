include ../../Makefile.defs

TESTNAME     = aomp-issue376
TESTSRC_MAIN = aomp-issue376.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

RUNENV       += OMPTEST_USE_OMPT_TRACING=1
RUNCMD       = ./$(TESTNAME) | $(FILECHECK) $(TESTSRC_MAIN)

CLANG        = clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE) $(OMPTEST)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
