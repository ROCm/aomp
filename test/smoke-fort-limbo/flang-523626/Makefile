# For testing without openmp:
# OMP_FLAGS    = -DNO_OMP
include ../../Makefile.defs

TESTNAME     = spread
TESTSRC_MAIN = spread.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
CFLAGS       = $(FLANG_GPU_LINK_FLAGS)
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
