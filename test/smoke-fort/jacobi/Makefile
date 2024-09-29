include ../../Makefile.defs

TESTNAME     = jacobi
TESTSRC_MAIN =
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang-new
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#OMP_FLAGS    =
#-ccc-print-phases
#"-\#\#\#"

#RUNENV       =
RUNCMD       = ./doit.sh ${TESTNAME} 2>&1 | tee ${TESTNAME}.perf

include ../Makefile.rules

jacobi:
	make FC=${OMP_BIN} ROCM_GPU=${AOMP_GPU} -f Makefile.jacobi
