include ../../Makefile.defs

TESTNAME     = collapse_parallel_spmd
TESTSRC_MAIN = collapse_parallel_spmd.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CFLAGS       += -Ofast -fno-openmp-target-ignore-env-vars
CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
