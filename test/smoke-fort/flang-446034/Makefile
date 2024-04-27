NOOPT        = 1
NOOMP        = 1
NOLINK       = 1
include ../../Makefile.defs
OMP_FLAGS    =

.PHONY: modules

TESTNAME     = spglib_f08.o
TESTSRC_MAIN = spglib_f08.F90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang-new
CFLAGS       = -c
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

RUNCMD       = ./doit.sh > ${TESTNAME}.out 2>&1 && ./chkit.sh ${TESTNAME}.out

include ../Makefile.rules

# 446005 "fails" version
spglib_f08.o:
	${OMP_BIN} -c spglib_version.f90 spglib_f08.F90
