include ../../Makefile.defs

TESTNAME        = function_template
TESTSRC_MAIN    = function_template.cpp
TESTSRC_AUX     =
TESTSRC_ALL     = $(TESTSRC_MAIN) $(TESTSRC_LIBA)
SCRIPT_INIT     =

CLANG           ?= clang++
OMP_BIN         = $(AOMP)/bin/$(CLANG)
CC              = $(OMP_BIN) $(VERBOSE)
EXTRA_LDFLAGS   =
EXTRA_OMP_FLAGS =

include ../Makefile.rules
