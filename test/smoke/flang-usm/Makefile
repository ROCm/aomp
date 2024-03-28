include ../../Makefile.defs
USE_OFFLOAD_ARCH = 1
HSA_XNACK        ?= 1
AOMP_TARGET_FEATURES = :xnack+

TESTNAME     = flang-usm
TESTSRC_MAIN = flang-usm.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

SUPPORTED    = $(SUPPORTS_USM)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)

RUNENV       += HSA_XNACK=$(HSA_XNACK)
RUNCMD       = ./$(TESTNAME)


#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
