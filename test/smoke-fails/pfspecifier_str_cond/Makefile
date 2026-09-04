include ../../Makefile.defs

TESTNAME        = pfspecifier_str_cond
TESTSRC_MAIN    = pfspecifier_str_cond.c
TESTSRC_AUX     =
TESTSRC_ALL     = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG           ?= clang
OMP_BIN         = $(AOMP)/bin/$(CLANG)
CC              = $(OMP_BIN) $(VERBOSE)
EXTRA_OMP_FLAGS =

include ../Makefile.rules
