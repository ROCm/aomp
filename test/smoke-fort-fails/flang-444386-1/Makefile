NOOPT        = 1
NOOMP        = 1
OMP_FLAGS    = -fopenmp
include ../../Makefile.defs

TESTNAME     = test_crayptr
TESTSRC_MAIN = test_crayptr.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang-new
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
