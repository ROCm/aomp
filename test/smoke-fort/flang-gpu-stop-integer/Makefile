include ../../Makefile.defs

TESTNAME     = flang-gpu-stop-integer
TESTSRC_MAIN = flang-gpu-stop-integer.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
CFLAGS       =
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)

RUNCMD       = ./doit.sh ./flang-gpu-stop-integer > ${TESTNAME}.out 2>&1 || ./chkit.sh ${TESTNAME}.out

include ../Makefile.rules
