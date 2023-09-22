include ../../Makefile.defs

TESTNAME     = omp_target_is
TESTSRC_MAIN = omp_target_is.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

ifeq ($(AOMP_GPU),gfx90a)
  HSA_XNACK=1
else
  HSA_XNACK=0
endif

CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
RUNENV       = HSA_XNACK=$(HSA_XNACK)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
