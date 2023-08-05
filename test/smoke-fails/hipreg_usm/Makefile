include ../../Makefile.defs

TESTNAME     = hostreg_usm
TESTSRC_MAIN = omp.cpp
TESTSRC_AUX  = hip_hostreg.o
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
AOMPHIP ?= $(AOMP)
HIPCC ?= $(AOMPHIP)/bin/hipcc

CFLAGS = -Ofast -fopenmp-target-fast
CLANG        = clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
EXTRA_CFLAGS = -L$(AOMPHIP)/lib -lamdhip64 -Wl,-rpath,$(AOMPHIP)/lib

HSA_XNACK ?= 1

#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
all: $(TESTNAME)

hip_hostreg.o : hip_hostreg.hip
	$(HIPCC) -c --offload-arch=$(AOMP_GPU) $^ -o $@

run:
	HSA_XNACK=${HSA_XNACK} ./$(TESTNAME)
