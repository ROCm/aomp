NOOPT        = 1
NOOMP        = 1
OMP_FLAGS    = -DNO_OMP
include ../../Makefile.defs

TESTNAME     = dtptr
TESTSRC_MAIN = dtptr.F90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang-new
CFLAGS       = -g -O3 -march=native
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

RUNCMD       = ./doit.sh ./dtptr 2>&1 | tee ${TESTNAME}.perf 2>&1 && ./chkit.sh ${TESTNAME}.perf

include ../Makefile.rules
