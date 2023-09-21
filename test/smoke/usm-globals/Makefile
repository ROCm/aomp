include ../../Makefile.defs

TESTNAME     = usm_globals
TESTSRC_MAIN = usm_globals.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
RUNENV      += OMPX_APU_MAPS=1 HSA_XNACK=1 LIBOMPTARGET_KERNEL_TRACE=2

CFLAGS       += -O3
CLANG        = clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
run: $(TESTNAME)
	$(RUNENV) ./$(TESTNAME) 2>&1 | $(AOMP)/bin/FileCheck $(TESTSRC_MAIN)
