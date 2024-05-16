include ../../Makefile.defs

TESTNAME     = atomic_vs_reduction
TESTSRC_MAIN = atomic_vs_reduction.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CFLAGS       += -fopenmp-target-fast
CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)

include ../Makefile.rules
