include ../../Makefile.defs

TESTNAME     = rush_larsen_openmp
TESTSRC_MAIN = rush_larsen_openmp.cc
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang -lm
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE) 
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
