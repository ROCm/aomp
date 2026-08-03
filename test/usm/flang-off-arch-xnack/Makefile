USE_OFFLOAD_ARCH = 1
ifeq ("$(AOMP_GPU)","gfx1250")
  AOMP_TARGET_FEATURES =
else ifeq ("$(AOMP_GPU)","gfx1251")
  AOMP_TARGET_FEATURES =
else
  AOMP_TARGET_FEATURES = :xnack-
endif

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
