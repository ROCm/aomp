include ../../Makefile.defs

TESTNAME     = generic-spmd-use-small-blocksize
TESTSRC_MAIN = generic-spmd-use-small-blocksize.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
RUNENV      += LIBOMPTARGET_KERNEL_TRACE=1
RUNENV 	    += LIBOMPTARGET_AMDGPU_GENERIC_SPMD_USE_SMALL_BLOCKSIZE=1

RUNCMD      = ./$(TESTNAME) 2>&1 | $(FILECHECK) $(TESTSRC_MAIN)

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
