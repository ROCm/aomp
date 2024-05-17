include ../../Makefile.defs

TESTNAME     = mapper_prob
TESTSRC_MAIN = mapper_prob.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE) 
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
