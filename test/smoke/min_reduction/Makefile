include ../../Makefile.defs

TESTNAME     = min_reduction
TESTSRC_MAIN = min_reduction.c
TESTSRC_ALL  = $(TESTSRC_MAIN)

CLANG        = clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)


UNSUPPORTED = ASAN_COMPILE

include ../Makefile.rules
