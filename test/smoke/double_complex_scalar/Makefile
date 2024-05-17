include ../../Makefile.defs
TESTNAME     = double_complex_scalar
TESTSRC_MAIN = double_complex_scalar.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE) -std=c++11 -latomic

include ../Makefile.rules
