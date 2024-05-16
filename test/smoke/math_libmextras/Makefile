include ../../Makefile.defs

TESTNAME        = math_libmextras
TESTSRC_MAIN    = math_libmextras.c
TESTSRC_AUX     =
TESTSRC_ALL     = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG           ?= clang
OMP_BIN         = $(AOMP)/bin/$(CLANG)
CC              = $(OMP_BIN) $(VERBOSE)
EXTRA_LDFLAGS   = -lm
EXTRA_OMP_FLAGS =

include ../Makefile.rules
