include ../../Makefile.defs

TESTNAME     = usm_locals
TESTSRC_MAIN = usm_locals.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
RUNENV      += HSA_XNACK=0 LIBOMPTARGET_KERNEL_TRACE=2

RUNCMD       = ./$(TESTNAME) 2>&1 | $(FILECHECK) $(TESTSRC_MAIN)

OMP_FLAGS    = -fopenmp --offload-arch=$(AOMP_GPU)
CFLAGS       += -O3
CLANG        = clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)

SUPPORTED    = $(SUPPORTS_USM)

#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
