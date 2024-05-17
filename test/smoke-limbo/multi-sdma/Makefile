include ../../Makefile.defs

TESTNAME     = multi-sdma
TESTSRC_MAIN = multi-sdma.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

RUNENV      = LIBOMPTARGET_AMDGPU_USE_MULTIPLE_SDMA_ENGINES=true
RUNENV     +=  LIBOMPTARGET_DEBUG=1
RUNCMD      = ./$(TESTNAME) 2>&1 | $(FILECHECK) $(TESTSRC_MAIN)

CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
