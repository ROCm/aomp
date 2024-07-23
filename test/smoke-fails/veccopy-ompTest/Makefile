include ../../Makefile.defs

TESTNAME     = veccopy-ompTest
TESTSRC_MAIN = veccopy.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
RUNENV       += OMPTEST_RUN_AS_TESTSUITE=1
CC           = $(OMP_BIN) $(VERBOSE) $(OMPTEST)

include ../Makefile.rules
