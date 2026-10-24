include ../../Makefile.defs

TESTNAME     = zero_copy
TESTSRC_MAIN = zero_copy.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
RUNENV      += HSA_XNACK=1 OMPX_DGPU_MAPS=0 LIBOMPTARGET_INFO=32

RUNCMD       = ./$(TESTNAME) 2>&1 | $(FILECHECK) $(TESTSRC_MAIN)

OMP_FLAGS    = -fopenmp --offload-arch=$(AOMP_GPU)
CFLAGS       += -O3
CLANG        = clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)

SUPPORTED    = gfx942

#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
