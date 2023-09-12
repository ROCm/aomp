NOOPT        = 1
NOOMP        = 1
OMP_FLAGS    = -DNO_OMP
include ../../Makefile.defs

TESTNAME     = abstract_type
TESTSRC_MAIN = abstract_type.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang-new
CFLAGS       = -flang-experimental-hlfir -flang-experimental-polymorphism
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
