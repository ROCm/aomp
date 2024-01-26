include ../../Makefile.defs

TESTNAME     = memset
TESTSRC_MAIN = memset.cpp
TESTSRC_AUX  = hip_memset.o
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
AOMPHIP ?= $(AOMP)
HIPCC ?= $(AOMPHIP)/bin/hipcc

CFLAGS = -O3
CLANG        = clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
EXTRA_CFLAGS = -L$(AOMPHIP)/lib -lamdhip64 -Wl,-rpath,$(AOMPHIP)/lib
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
all: $(TESTNAME)

hip_memset.o : hip_memset.hip
	$(HIPCC) -c --offload-arch=$(AOMP_GPU) $^ -o $@
