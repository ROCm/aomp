include ../../Makefile.defs

TESTNAME     = flang-gpu-abort
TESTSRC_MAIN = flang-gpu-abort.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang-new
CFLAGS       =
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)

RUNCMD       =  ./runit.sh > ${TESTNAME}.out 2>&1 && exit 1 || ./chkit.sh ${TESTNAME}.out

include ../Makefile.rules
