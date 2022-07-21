include ../../Makefile.defs

TESTNAME     = veccopy-ompt-target-disallow-both
TESTSRC_MAIN = veccopy-ompt-target-disallow-both.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
run:
	./$(TESTNAME) | $(AOMP)/bin/FileCheck $(TESTSRC_MAIN)
