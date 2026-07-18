#NOOPT        = 1
#NOOMP        = 1
#OMP_FLAGS    = -fopenmp
include ../../Makefile.defs

TESTNAME     = device_intrinsics_map
TESTSRC_MAIN = device_intrinsics_map.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
CFLAGS       = $(FLANG_GPU_LINK_FLAGS)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
