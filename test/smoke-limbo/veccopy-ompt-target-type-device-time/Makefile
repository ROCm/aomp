include ../../Makefile.defs

TESTNAME     = veccopy-ompt-target
TESTSRC_MAIN = veccopy-ompt-target.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
run:
	echo $(PATH)
	./$(TESTNAME) | $(FILECHECK) $(TESTSRC_MAIN)
