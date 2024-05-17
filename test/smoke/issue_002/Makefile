include ../../Makefile.defs

TESTNAME     = issue_002
TESTSRC_MAIN = issue_002.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
EXTRA_CFLAGS += -g

include ../Makefile.rules
