NOOPT        = 1
include ../../Makefile.defs

TESTNAME     = saxpy-2d-device
TESTSRC_MAIN = saxpy-2d-device.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
CFLAGS       =
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
OMP_FLAGS   += -fdo-concurrent-to-openmp=device

#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
