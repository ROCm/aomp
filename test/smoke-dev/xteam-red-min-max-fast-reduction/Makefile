include ../../Makefile.defs

TESTNAME     = xteam-red-min-max-fast-reduction
TESTSRC_MAIN = xteam-red-min-max-fast-reduction.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CFLAGS      += -fopenmp-target-fast-reduction
EXTRA_LDFLAGS     += -lm
CLANG        = clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
