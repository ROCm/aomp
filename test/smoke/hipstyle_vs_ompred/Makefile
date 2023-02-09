include ../../Makefile.defs

TESTNAME     = hipstyle_vs_ompred
TESTSRC_MAIN = hipstyle_vs_ompred.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CFLAGS       += -fopenmp-target-fast
CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)

include ../Makefile.rules
