include ../../Makefile.defs

TESTNAME     = clang-434259
TESTSRC_MAIN = clang-434259.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
OMP_FLAGS 	 += -O3 --offload-arch=gfx942 -fopenmp-version=51 -D__HIP_PLATFORM_AMD__ -DUSE_DEVICE_ALLOC__ -DUSE_DEVICE_ALWAYS

SUPPORTED    = $(SUPPORTS_USM)

#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
