include ../../Makefile.defs

TESTNAME     = clang-reduce
TESTSRC_MAIN = 
TESTSRC_AUX  = clang-reduce.c
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
