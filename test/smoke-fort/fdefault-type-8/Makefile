NOOPT        = 1
NOOMP        = 1
# Note: -fdefault-double-8 needed
OMP_FLAGS    = -fdefault-integer-8 -fdefault-real-8 -fdefault-double-8
include ../../Makefile.defs

TESTNAME     = daxpy_example
TESTSRC_MAIN = daxpy_example.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang-new
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

RUNCMD      = ./$(TESTNAME) 2>&1 | diff - expected.out

include ../Makefile.rules
