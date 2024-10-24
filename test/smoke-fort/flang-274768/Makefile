NOOPT        = 1
#NOOMP        = 1
#OMP_FLAGS    = -fopenmp
include ../../Makefile.defs

TESTNAME     = assumed
TESTSRC_MAIN = assumed.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang-new
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

RUNCMD       = ./${TESTNAME} 2>&1 | tee ${TESTNAME}.out 2>&1 && diff ${TESTNAME}.out chk.stdout

include ../Makefile.rules
