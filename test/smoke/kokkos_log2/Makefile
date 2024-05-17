include ../../Makefile.defs

TESTNAME        = kokkos_log2
TESTSRC_MAIN    = kokkos_log2.cpp
TESTSRC_AUX     =
TESTSRC_ALL     = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG           ?= clang++
OMP_BIN         = $(AOMP)/bin/$(CLANG)
CC              = $(OMP_BIN) $(VERBOSE)
EXTRA_LDFLAGS   =
EXTRA_OMP_FLAGS =

include ../Makefile.rules
