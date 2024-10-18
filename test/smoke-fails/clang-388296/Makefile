include ../../Makefile.defs

TESTNAME     = clang-388296
TESTSRC_MAIN = clang-388296.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang++  --std=c++17  -fnew-alignment=256  -D__HIP_PLATFORM_AMD__  -fopenmp-version=51 -DUSE_UNIFIED_MEMORY_MINMAX -L${AOMPHIP}/lib -lamdhip64

OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)

HSA_XNACK ?= 1

SUPPORTED    = $(SUPPORTS_USM)

#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
run: $(TESTNAME)
	        HSA_XNACK=${HSA_XNACK} ./$(TESTNAME)

