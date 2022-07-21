include ../../Makefile.defs

TESTNAME     = no_loop_3
TESTSRC_MAIN = no_loop_3.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
RUNENV      += LIBOMPTARGET_KERNEL_TRACE=1

CFLAGS       += -O3
CLANG        = clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
run: $(TESTNAME)
	$(RUNENV) ./$(TESTNAME) 2>&1 | $(AOMP)/bin/FileCheck $(TESTSRC_MAIN)
