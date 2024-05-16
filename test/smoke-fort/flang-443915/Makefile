NOOPT        = 1
NOOMP        = 1
OMP_FLAGS    = -fopenmp
include ../../Makefile.defs

TESTNAME     = label_and_default
TESTSRC_MAIN = label_and_default.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang-new
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
