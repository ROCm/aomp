include ../../Makefile.defs

TESTNAME     = imperfect_loop_collapse
TESTSRC_MAIN = imperfect_loop_collapse.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)

#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules

run:
	./$(TESTNAME)
