NOOPT        = 1
NOOMP        = 1
NOLINK       = 1
include ../../Makefile.defs
OMP_FLAGS    =

.PHONY: modules

TESTNAME     = impl.o
TESTSRC_MAIN = impl.f90
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

# 446005 "works" version
impl.o:
	${OMP_BIN} -c interface.f90
	${OMP_BIN} -c impl.f90
