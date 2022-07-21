include ../../Makefile.defs

TESTNAME     = omp_num_teams_generic
TESTSRC_MAIN = omp_num_teams_generic.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG) -isystem$(AOMP)/include
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
