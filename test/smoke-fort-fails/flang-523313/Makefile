#NOOPT        = 1
#NOOMP        = 1
#OMP_FLAGS    = -fopenmp
include ../../Makefile.defs

TESTNAME     = repro_derived
TESTSRC_MAIN = repro_derived.F90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
