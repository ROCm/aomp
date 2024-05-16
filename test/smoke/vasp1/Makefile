include ../../Makefile.defs

TESTNAME     = vasp1
TESTSRC_MAIN = vasp1.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
RUNENV      += LIBOMPTARGET_KERNEL_TRACE=1

CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"
include ../Makefile.rules

run: $(TESTNAME)
	$(RUNENV) ./$(TESTNAME) 1

