include ../../Makefile.defs

TESTNAME     = milestone-3-babel-noteams
TESTSRC_MAIN =
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang-new
OMP_BIN      =
CC           = ./buildit.sh
#-ccc-print-phases
#"-\#\#\#"

RUNCMD       = ./doit.sh 2>&1 | tee ${TESTNAME}.perf 2>&1 && ./chkit.sh ${TESTNAME}.perf

include ../Makefile.rules
