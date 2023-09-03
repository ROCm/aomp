include ../../Makefile.defs

TESTNAME     = clang-370300
TESTSRC_MAIN = 
TESTSRC_AUX  = clang-370300.c
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
