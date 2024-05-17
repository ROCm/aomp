include ../../Makefile.defs

TESTNAME     = clang_llnl-58
TESTSRC_MAIN = clang_llnl_58.cc
TESTSRC_AUX  = 
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

OMP_FLAGS =  -c -D__HIP_PLATFORM_AMD__ -I$(AOMPHIP)/include -O3 -g -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908

CLANG        = clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
