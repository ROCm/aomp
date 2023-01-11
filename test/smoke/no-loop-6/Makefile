include ../../Makefile.defs

TESTNAME     = no_loop_6
TESTSRC_MAIN = no_loop_6.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
RUNENV      += LIBOMPTARGET_KERNEL_TRACE=1

TESTCMD      = ./$(TESTNAME) 2>&1 | $(AOMP)/bin/FileCheck $(TESTSRC_MAIN)

CFLAGS       += -Ofast
CLANG        = clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
run: $(TESTNAME)
	$(RUNENV) ./$(TESTNAME) 2>&1 | $(AOMP)/bin/FileCheck $(TESTSRC_MAIN)
