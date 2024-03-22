USE_OFFLOAD_ARCH = 1

include ../../Makefile.defs

TESTNAME     = flang-gpu-check
TESTSRC_MAIN = flang-gpu-check.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        = flang 
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
