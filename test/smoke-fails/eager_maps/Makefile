include ../../Makefile.defs

TESTNAME     = eager_maps
TESTSRC_MAIN = eager_maps.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
RUNENV      += HSA_XNACK=1 OMPX_EAGER_ZERO_COPY_MAPS=1 LIBOMPTARGET_KERNEL_TRACE=2

RUNCMD       = ./$(TESTNAME) 2>&1 | $(FILECHECK) $(TESTSRC_MAIN)

OMP_FLAGS    = -fopenmp --offload-arch=$(AOMP_GPU)
CFLAGS       += -O3
CLANG        = clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)

ifeq ($(AOMP_IS_APU),1)
SUPPORTED    = gfx940,gfx941,gfx942
else
SUPPORTED    = gfx940
endif

#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
