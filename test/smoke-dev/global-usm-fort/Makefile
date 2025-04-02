include ../../Makefile.defs

TESTNAME     = global-usm-fort
TESTSRC_MAIN =
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
OMP_BIN      =
CC           = AOMP=${AOMP} AOMP_GPU=${AOMP_GPU} FLANG=${FLANG} TESTNAME=${TESTNAME} ./wrapper.sh
#-ccc-print-phases
#"-\#\#\#"

RUNENV       = AOMP_GPU=${AOMP_GPU}
RUNCMD       = ./chkit.sh ${TESTNAME}.log

include ../Makefile.rules
