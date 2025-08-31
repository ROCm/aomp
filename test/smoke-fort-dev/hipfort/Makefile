#NOOPT        = 1
#NOOMP        = 1
#OMP_FLAGS    = -fopenmp
include ../../Makefile.defs

TESTNAME     = hipfort
TESTSRC_MAIN =
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
HIP_LIB_PATH ?= $(realpath $(realpath ${AOMP})/../../lib)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules

PATH        := ${AOMP}/../../bin:${AOMP}/../bin:${AOMP}/bin:${PATH}
LD_LIBRARY_PATH := ${HIP_LIB_PATH}:${LD_LIBRARY_PATH}
RUNCMD       = echo "Done hipfort"

.PHONY: hipfort
hipfort:
	echo PATH=${PATH}
	echo LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
	make -f Make.hipfort clean_all
	env PATH=${PATH} make -f Make.hipfort build_all
	env LD_LIBRARY_PATH=${LD_LIBRARY_PATH} make -f Make.hipfort run_all
	echo "Done Make.hipfort"

clean::
	make -f Make.hipfort clean_all
