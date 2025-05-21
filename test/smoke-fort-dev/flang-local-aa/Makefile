include ../../Makefile.defs

TESTNAME     = flang-local-aa
TESTSRC_MAIN = flang-local-aa.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
OPT_FLAGS    = -O3
FLANG        ?= flang
CFLAGS       =
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE) $(OPT_FLAGS)

#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
