include ../../Makefile.defs

TESTNAME     = FindPrimes
TESTSRC_MAIN = FindPrimes.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)


CLANG        ?= clang -lm
# CFLAGS       += -fopenmp-target-fast -lm
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)

include ../Makefile.rules

