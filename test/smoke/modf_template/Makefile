include ../../Makefile.defs

TESTNAME        = modf_template
TESTSRC_MAIN    = modf_template.cpp
TESTSRC_AUX     =
TESTSRC_ALL     = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG           ?= clang++
OMP_BIN         = $(AOMP)/bin/$(CLANG)
CC              = $(OMP_BIN) $(VERBOSE)

include ../Makefile.rules
