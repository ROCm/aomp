include ../../Makefile.defs
TESTNAME     = complex_reduction
TESTSRC_MAIN = complex_reduction.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)

UNSUPPORTED = ASAN_COMPILE

include ../Makefile.rules
