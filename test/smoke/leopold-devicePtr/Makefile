include ../../Makefile.defs

TESTNAME     = leopold-devicePtr
TESTSRC_MAIN = leopold-devicePtr.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

HSA_XNACK    ?= 1

CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
RUNENV       = HSA_XNACK=$(HSA_XNACK)

SUPPORTED = gfx90a

#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
