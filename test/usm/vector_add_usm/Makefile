include ../../Makefile.defs

TESTNAME     = vector_add_usm
TESTSRC_MAIN = vector_add_usm.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
RUNENV      += HSA_XNACK=1

OMP_FLAGS    = -fopenmp --offload-arch=$(AOMP_GPU)
CFLAGS       += -O3
CLANG        = clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)

SUPPORTED    = gfx90a,gfx940,gfx941,gfx942

#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
