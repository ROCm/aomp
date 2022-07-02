include ../../Makefile.defs

ROCM         ?= $(AOMP)/
TESTNAME     = flang-collapseNR
TESTSRC_MAIN = flang-collapseNR.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_AUX) $(TESTSRC_MAIN)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
