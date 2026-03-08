NOOPT        = 1
NOOMP        = 0
SUPPORTED    = $(SUPPORTS_USM)
OMP_FLAGS    = -fopenmp --offload-arch=$(AOMP_GPU)
include ../../Makefile.defs

TESTNAME     = test
TESTSRC_MAIN = test.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CFLAGS       = -ffast-amd-memory-allocator -lflang_rt.amd
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
