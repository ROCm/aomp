include ../../Makefile.defs

TESTNAME     = collapse1
TESTSRC_MAIN = collapse1.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

RUNENV      += LIBOMPTARGET_KERNEL_TRACE=1
RUNCMD      = ./$(TESTNAME) 2>&1 | $(FILECHECK) $(TESTSRC_MAIN)

CFLAGS       += -O3 -fopenmp-target-fast
CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
