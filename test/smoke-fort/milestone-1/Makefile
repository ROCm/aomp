include ../../Makefile.defs

TESTNAME     = main
TESTSRC_MAIN = main.f95
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang-new
OMP_BIN      = $(AOMP)/bin/$(FLANG) -Rpass=openmp-opt -Rpass-missed=openmp-opt -Rpass-analysis=openmp-opt -Rpass-missed=loop-vectorize
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
