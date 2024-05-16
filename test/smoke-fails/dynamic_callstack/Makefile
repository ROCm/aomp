include ../../Makefile.defs

TESTNAME     = dynamic_callstack
TESTSRC_MAIN = dynamic_callstack.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules

run:
	OMP_TARGET_OFFLOAD=MANDATORY LIBOMPTARGET_STACK_SIZE=4096 ./$(TESTNAME)
