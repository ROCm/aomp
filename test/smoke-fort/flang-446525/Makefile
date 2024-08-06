NOOPT        = 1
include ../../Makefile.defs
OMP_FLAGS    += -O3

TESTNAME     = rush_larsen_gpu_omp_fort
TESTSRC_MAIN = rush_larsen_gpu_omp_fort.F90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang-new
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

RUNENV       = AOMP_GPU=${AOMP_GPU}
RUNCMD       = ./doit.sh 2>&1 | tee ${TESTNAME}.perf 2>&1 && ./chkit.sh ${TESTNAME}.perf

include ../Makefile.rules
