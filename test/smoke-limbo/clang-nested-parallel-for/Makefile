include ../../Makefile.defs

TESTNAME     = clang-nested-parallel-for
TESTSRC_MAIN = clang-nested-parallel-for.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
OMP_FLAGS += -O0
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
