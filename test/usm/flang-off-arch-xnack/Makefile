USE_OFFLOAD_ARCH = 1
AOMP_TARGET_FEATURES=:xnack-

include ../../Makefile.defs

TESTNAME     = flang-off-arch-xnack
TESTSRC_MAIN = flang-off-arch-xnack.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        = flang 
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
run:
	timeout 10s ./flang-off-arch-xnack 
