include ../../Makefile.defs

TESTNAME     = jacobi
TESTSRC_MAIN =
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang-new
OMP_BIN      =
CC           = AOMP=${AOMP} AOMP_GPU=${AOMP_GPU} FLANG=${FLANG} ./buildit.sh
#-ccc-print-phases
#"-\#\#\#"

RUNENV       = AOMP_GPU=${AOMP_GPU}
RUNCMD       = ./doit.sh 2>&1 | tee ${TESTNAME}.perf 2>&1 && ./chkit.sh ${TESTNAME}.perf

include ../Makefile.rules
