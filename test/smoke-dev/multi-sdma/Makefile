include ../../Makefile.defs

TESTNAME     = multi-sdma
TESTSRC_MAIN = multi-sdma.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

RUNENV      = LIBOMPTARGET_AMDGPU_USE_MULTIPLE_SDMA_ENGINES=true
RUNENV     +=  LIBOMPTARGET_DEBUG=1
RUNCMD      = ./$(TESTNAME) 2>&1 | awk -f round-robin.awk | $(FILECHECK) $(TESTSRC_MAIN)

UNSUPPORTED = gfx1151

CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
