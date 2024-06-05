include ../../Makefile.defs

TESTNAME     = no_loop_7
TESTSRC_MAIN = no_loop_7.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
RUNENV      += LIBOMPTARGET_KERNEL_TRACE=1

RUNCMD      = ./$(TESTNAME) 2>&1 | $(FILECHECK) $(TESTSRC_MAIN)

CFLAGS       += -Ofast -fno-openmp-target-ignore-env-vars
CLANG        = clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
