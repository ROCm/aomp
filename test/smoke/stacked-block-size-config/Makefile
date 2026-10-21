include ../../Makefile.defs

TESTNAME     = stacked_block_size_config
TESTSRC_MAIN = stacked_block_size_config.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
RUNENV      += LIBOMPTARGET_MIN_THREADS_FOR_LOW_TRIP_COUNT=16 LIBOMPTARGET_KERNEL_TRACE=1

RUNCMD       = ./$(TESTNAME) 2>&1 | $(FILECHECK) $(TESTSRC_MAIN)

# The low trip count thread reduction applies to plain SPMD kernels.
CFLAGS       += -O3 -fopenmp-gpu-threads-per-team=128 -fno-openmp-target-big-jump-loop
CLANG        = clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)

#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules

