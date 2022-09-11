include ../../Makefile.defs

TESTNAME     = flang-349966
TESTSRC_MAIN = flang-349966.f90
TESTSRC_AUX  = tools.f90
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG) -O3 -ffast-math
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
run:
	ulimit -s unlimited; ./$(TESTNAME) < w0008_631g.inp
