include ../../Makefile.defs

TESTNAME     = fdefault-sizeof
TESTSRC_MAIN =
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang-new
OMP_BIN      =
CC           = AOMP=${AOMP} AOMP_GPU=${AOMP_GPU} FLANG=${FLANG} ./buildit.sh
#-ccc-print-phases
#"-\#\#\#"

RUNENV       = AOMP=${AOMP}
RUNCMD       = ./doit.sh > ${TESTNAME}.out 2>&1 && ./chkit.sh ${TESTNAME}.out

include ../Makefile.rules
