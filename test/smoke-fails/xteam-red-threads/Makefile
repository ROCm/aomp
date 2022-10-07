include ../../Makefile.defs

TESTNAME     = xteam_red_threads
TESTSRC_MAIN = xteam_red_threads.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
RUNENV      += LIBOMPTARGET_KERNEL_TRACE=1

CFLAGS       += -fopenmp-target-ignore-env-vars
CLANG        = clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
run: $(TESTNAME)
	$(RUNENV) ./$(TESTNAME) 2>&1 | $(AOMP)/bin/FileCheck $(TESTSRC_MAIN)
