#NOOPT        = 1
#NOOMP        = 1
#OMP_FLAGS    = -DNO_OMP
include ../../Makefile.defs

TESTNAME     = minimal
TESTSRC_MAIN = minimal_whole_array_ops.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
CFLAGS       = -fopenmp-target-fast -fopenmp-assume-threads-oversubscription -fopenmp-assume-teams-oversubscription 
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

RUNCMD       = ./doit.sh ${TESTNAME} && ./chkit.sh ${TESTNAME}

include ../Makefile.rules
