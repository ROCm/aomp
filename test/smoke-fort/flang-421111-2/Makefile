NOOPT        = 1
NOOMP        = 1
OMP_FLAGS    = -DNO_OMP
include ../../Makefile.defs

TESTNAME     = pointer_intent_2
TESTSRC_MAIN = pointer_intent_2.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang-new
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
