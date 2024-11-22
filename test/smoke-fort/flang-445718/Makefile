NOOPT        = 1
include ../../Makefile.defs

TESTNAME     = rush_larsen_gpu_omp_fort_noprivate
TESTSRC_MAIN = rush_larsen_gpu_omp_fort_noprivate.F90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang-new
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

RUNCMD = ./$(TESTNAME) 100 1

include ../Makefile.rules
