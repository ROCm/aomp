include ../../Makefile.defs

TESTNAME     = test
TESTSRC_MAIN = test.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang
OMP_BIN       = $(AOMP)/bin/$(CLANG)
CC            = $(OMP_BIN) $(VERBOSE)
OMP_FLAGS    += -O3 
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
