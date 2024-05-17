include ../../Makefile.defs

TESTNAME     = map_to_from_prob
TESTSRC_MAIN = map_to_from_prob.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)

# RUNENV = LD_LIBRARY_PATH=$(AOMP)/lib-debug #LIBOMPTARGET_DEBUG=1
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
