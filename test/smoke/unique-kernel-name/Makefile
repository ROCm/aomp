include ../../Makefile.defs

TESTNAME        = unique-kernel-name
TESTSRC_MAIN    = b.cpp
TESTSRC_AUX     = a.cpp
TESTSRC_ALL     = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG           ?= clang++
OMP_BIN         = $(AOMP)/bin/$(CLANG)
CC              = $(OMP_BIN) $(VERBOSE)
EXTRA_LDFLAGS   =
EXTRA_OMP_FLAGS =

include ../Makefile.rules
