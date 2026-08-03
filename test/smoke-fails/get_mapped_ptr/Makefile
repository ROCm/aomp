include ../../Makefile.defs
HSA_XNACK        ?= 1
ifeq ("$(AOMP_GPU)","gfx1250")
  AOMP_TARGET_FEATURES =
else ifeq ("$(AOMP_GPU)","gfx1251")
  AOMP_TARGET_FEATURES =
else
  AOMP_TARGET_FEATURES = :xnack+
endif

TESTNAME     = get_mapped_ptr
TESTSRC_MAIN = get_mapped_ptr.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

SUPPORTED    = $(SUPPORTS_USM)

CLANG        ?= clang++ -fopenmp-version=52
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

RUNENV       += HSA_XNACK=$(HSA_XNACK)
RUNCMD       = ./$(TESTNAME)


include ../Makefile.rules
