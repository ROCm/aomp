include ../../Makefile.defs

TESTNAME     = clang_llnl-58
TESTSRC_MAIN = clang_llnl-58.cc
TESTSRC_AUX  = 
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
