include ../../Makefile.defs

TESTNAME     = clang-stack-prob
TESTSRC_MAIN = 
TESTSRC_AUX  = clang-stack-prob.c
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
