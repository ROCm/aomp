include ../../Makefile.defs

TESTNAME     = flang-gpu-exit
TESTSRC_MAIN = flang-gpu-exit.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang-new
CFLAGS       =
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)

RUNCMD       = ./flang-gpu-exit > ${TESTNAME}.out 2>&1 && exit 1 || ./chkit.sh ${TESTNAME}.out

include ../Makefile.rules
