#NOOPT        = 1
#NOOMP        = 1
#OMP_FLAGS    = -fopenmp
include ../../Makefile.defs

TESTNAME     = hipfort
TESTSRC_MAIN =
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang-new
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules

PATH        := ${AOMP}/../../bin:${AOMP}/../bin:${AOMP}/bin:${PATH}
RUNCMD       = echo "Done hipfort"

.PHONY: hipfort
hipfort:
	make -f Make.hipfort clean_all
	env PATH=${PATH} make -f Make.hipfort build_all
	make -f Make.hipfort run_all
	echo "Done Make.hipfort"

clean::
	make -f Make.hipfort clean_all
