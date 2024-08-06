include ../../Makefile.defs

TESTNAME     = matmul_parallel_spmd
TESTSRC_MAIN = matmul_parallel_spmd.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CFLAGS       += -O3 -fno-openmp-target-ignore-env-vars
CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
