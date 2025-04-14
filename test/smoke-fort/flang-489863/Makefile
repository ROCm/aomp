NOOPT        = 1
include ../../Makefile.defs

TESTNAME     = omp
TESTSRC_MAIN = omp.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
CFLAGS       = -O3 -fopenmp-target-fast
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"
export TIMEOUT = 120s

RUNCMD       = ./doit.sh ./${TESTNAME} 2>&1 | tee ${TESTNAME}.perf

include ../Makefile.rules
