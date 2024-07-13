include ../../Makefile.defs

TESTNAME     = flang_deviceio
TESTSRC_MAIN = flang_deviceio.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

TARGET       = -fopenmp -O3 -w -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$(AOMP_GPU)$(AOMP_TARGET_FEATURES)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

UNSUPPORTED = ASAN_COMPILE

include ../Makefile.rules
