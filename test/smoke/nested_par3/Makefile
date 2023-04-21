include ../../Makefile.defs

TESTNAME     = nested_par3
TESTSRC_MAIN = nested_par3.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE) -mllvm -openmp-opt-max-iterations=4  
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
