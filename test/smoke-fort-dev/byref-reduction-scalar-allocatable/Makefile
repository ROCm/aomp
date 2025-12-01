NOOPT        = 1
include ../../Makefile.defs

TESTNAME     = byref_reduction_scalar_allocatable
TESTSRC_MAIN = byref_reduction_scalar_allocatable.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN)

FLANG        ?= flang
CFLAGS       = $(FLANG_GPU_LINK_FLAGS)
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
