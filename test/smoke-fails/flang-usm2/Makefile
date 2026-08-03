USE_OFFLOAD_ARCH = 1
ifeq ("$(AOMP_GPU)","gfx1250")
  AOMP_TARGET_FEATURES =
else ifeq ("$(AOMP_GPU)","gfx1251")
  AOMP_TARGET_FEATURES =
else
  AOMP_TARGET_FEATURES = :xnack+
endif

include ../../Makefile.defs

TESTNAME     = flang-usm
TESTSRC_MAIN = flang-usm.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        = flang 
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)

SUPPORTED    = $(SUPPORTS_USM)

#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
