include ../../Makefile.defs

TESTNAME     = iteam_red_1
TESTSRC_MAIN = iteam_red_1.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CFLAGS       += -O3 -fopenmp-assume-no-nested-parallelism -fopenmp-assume-no-thread-state
CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
