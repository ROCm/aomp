include ../../Makefile.defs

TESTNAME     = occupancy-based-opt-generic-spmd-use-small-blocksize
TESTSRC_MAIN = occupancy-based-opt-generic-spmd-use-small-blocksize.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
RUNENV      += LIBOMPTARGET_KERNEL_TRACE=1
RUNENV 	    += OMPX_GENERIC_SPMD_OCCUPANCY_BASED_OPT=1

RUNCMD      = ./$(TESTNAME) 2>&1 | $(FILECHECK) $(TESTSRC_MAIN)

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
