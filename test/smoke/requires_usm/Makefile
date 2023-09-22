include ../../Makefile.defs

TESTNAME     = requires_usm
TESTSRC_MAIN = requires_usm.cpp
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

SUPPORTED = gfx90a

#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
