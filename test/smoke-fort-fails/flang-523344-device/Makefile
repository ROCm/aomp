#NOOPT        = 1
#NOOMP        = 1
#OMP_FLAGS    = -fopenmp
include ../../Makefile.defs

TESTNAME     = saxpy_gpu_do_concurrent_device
TESTSRC_MAIN = saxpy_gpu_do_concurrent_device.F90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
CFLAGS       = -fdo-concurrent-parallel=device
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
