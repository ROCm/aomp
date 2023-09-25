include ../../Makefile.defs

TESTNAME     = nested_loops_timing
TESTSRC_MAIN = nested_loops_timing.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CFLAGS       += -fopenmp-target-fast
CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)

include ../Makefile.rules
