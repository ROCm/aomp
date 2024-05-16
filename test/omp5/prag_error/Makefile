include ../../Makefile.defs

TESTNAME     = prag_error
TESTSRC_MAIN = prag_error.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE) 
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
