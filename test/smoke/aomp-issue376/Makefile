include ../../Makefile.defs

TESTNAME     = aomp-issue376
TESTSRC_MAIN = aomp-issue376.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

TESTCMD      = ./$(TESTNAME) | $(AOMP)/bin/FileCheck $(TESTSRC_MAIN)

CLANG        = clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
run:
	./$(TESTNAME) | $(AOMP)/bin/FileCheck $(TESTSRC_MAIN)
