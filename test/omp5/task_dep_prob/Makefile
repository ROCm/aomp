include ../../Makefile.defs

TESTNAME     = task_dep_prob
TESTSRC_MAIN = task_dep_prob.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE) 
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
