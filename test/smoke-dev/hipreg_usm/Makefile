include ../../Makefile.defs
HSA_XNACK ?= 1
AOMP_TARGET_FEATURES = :xnack+

TESTNAME     = hostreg_usm
TESTSRC_MAIN = omp.cpp
TESTSRC_AUX  = hip_hostreg.o
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
AOMPHIP ?= $(AOMP)
HIPCC ?= $(AOMPHIP)/bin/hipcc
HIP_CLANG_PATH ?= $(AOMPHIP)/bin

CFLAGS = -O3 -fopenmp-target-fast
CLANG        = clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
EXTRA_CFLAGS = -L$(AOMPHIP)/lib -lamdhip64 -Wl,-rpath,$(AOMPHIP)/lib

SUPPORTED    = $(SUPPORTS_USM)


#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
all: $(TESTNAME)

hip_hostreg.o : hip_hostreg.hip
	HIP_CLANG_PATH=$(HIP_CLANG_PATH) $(HIPCC) -c --offload-arch=$(AOMP_GPU) $^ -o $@

RUNENV       += HSA_XNACK=$(HSA_XNACK)
RUNCMD       = ./$(TESTNAME)

