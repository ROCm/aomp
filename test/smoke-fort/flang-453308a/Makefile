include ../../Makefile.defs

TESTNAME     = matsol.o
TESTSRC_MAIN = matsol.f
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang-new
CFLAGS       = -c -O3 -Dundef -DPERF 
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

RUNCMD       = ./doit.sh > ${TESTNAME}.out 2>&1 && ./chkit.sh ${TESTNAME}.out

include ../Makefile.rules
