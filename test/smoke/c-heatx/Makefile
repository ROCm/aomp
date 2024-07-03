include ../../Makefile.defs

TESTNAME     = c-heatx
TESTSRC_MAIN = c-heatx.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang -fopenmp-target-ignore-env-vars
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

UNSUPPORTED = ASAN_COMPILE

include ../Makefile.rules
