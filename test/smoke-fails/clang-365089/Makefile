include ../../Makefile.defs

TESTNAME     = clang-365089
TESTSRC_MAIN = 
TESTSRC_AUX  = clang-365089.cc
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)

OMP_FLAGS = -fopenmp -Ofast -v -c

include ../Makefile.rules
