include ../../Makefile.defs

TESTNAME     = occupancy_based_opt_big_jump_loop
TESTSRC_MAIN = occupancy_based_opt_big_jump_loop.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
RUNENV      += LIBOMPTARGET_KERNEL_TRACE=1
RUNENV 		+= OMPX_BIGJUMPLOOP_OCCUPANCY_BASED_OPT=1

RUNCMD      = ./$(TESTNAME) 2>&1 | $(FILECHECK) $(TESTSRC_MAIN)

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
