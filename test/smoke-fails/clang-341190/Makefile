include ../../Makefile.defs

TESTNAME     = clang-341190
TESTSRC_MAIN = clang-341190.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang++
OMP_FLAGS	+=  -DUSE_USM -march=znver3 -O3 -std=c++17 -D__HIP_PLATFORM_AMD__ -O3 -L${AOMPHIP}/lib -lamdhip64 -I${AOMPHIP}/include/hip
#(export ROCM_PATH=~/rocm/aomp_15.0-18/ ; ~/rocm/aomp_15.0-18/bin/clang++ -DUSE_USM -march=znver3 -O3 -std=c++17 -fopenmp -D__HIP_PLATFORM_AMD__ -O3 -fopenmp -target x86_64-pc-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a --offload-arch=gfx90a clang-341190.cpp -L${ROCM_PATH}/lib -lamdhip64 -I${ROCM_PATH}/include/hip -o hmm_rocm)

OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)

HSA_XNACK ?= 1

SUPPORTED    = $(SUPPORTS_USM)

#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules

run:
	HSA_XNACK=${HSA_XNACK} ./${TESTNAME}
