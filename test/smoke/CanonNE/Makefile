include ../../Makefile.defs

TESTNAME     = CanonNE
TESTSRC_MAIN = CanonNE.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG) -isystem$(AOMP)/include
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
