USE_OFFLOAD_ARCH = 1
AOMP_TARGET_FEATURES=:xnack-

include ../../Makefile.defs

TESTNAME     = flang-usm-235
TESTSRC_MAIN = flang-usm.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        = flang 
OMP_BIN      = $(AOMP)/bin/$(FLANG)  -Mx,235,1
CC           = $(OMP_BIN) $(VERBOSE)

SUPPORTED    = $(SUPPORTS_USM)

#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
