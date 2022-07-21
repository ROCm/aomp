include ../../Makefile.defs

TESTNAME     = reduc_map_prob
TESTSRC_MAIN = reduc_map_prob.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE) -fopenmp-version=50
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
