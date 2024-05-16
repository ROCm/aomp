include ../../Makefile.defs
TESTNAME     = omp_get_initial
TESTSRC_MAIN = omp_get_initial.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)

include ../Makefile.rules
