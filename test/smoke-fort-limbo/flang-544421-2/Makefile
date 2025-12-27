#NOOPT        = 1
#NOOMP        = 1
include ../../Makefile.defs

# Required for the test to succeed at the moment, but also
# required to hit the original problem.
RUNENV       += HSA_XNACK=1
OMP_FLAGS    = -fopenmp-force-usm --offload-arch=$(AOMP_GPU)

TESTNAME     = repro_544421-2
TESTSRC_MAIN = repro_544421-2.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
