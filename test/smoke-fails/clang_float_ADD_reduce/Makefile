include ../../Makefile.defs

TESTNAME     = clang-reduceADD
TESTSRC_MAIN = 
TESTSRC_AUX  = clang-reduceADD.cpp
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
