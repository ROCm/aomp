include ../../Makefile.defs

TESTNAME     = use-dev-addr-performance
TESTSRC_MAIN = main.f90
TESTSRC_AUX  = bar.o
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
AOMPHIP ?= $(AOMP)
HIPCC ?= $(AOMPHIP)/bin/hipcc
HIP_CLANG_PATH ?= $(AOMP)/bin

CFLAGS = -O3
FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
EXTRA_CFLAGS = -L$(AOMPHIP)/lib -lamdhip64 -Wl,-rpath,$(AOMPHIP)/lib -fPIC
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
all: $(TESTNAME)

bar.o : bar.hip
	HIP_CLANG_PATH=$(HIP_CLANG_PATH) $(HIPCC) -c --offload-arch=$(AOMP_GPU) -fPIC $^ -o $@

run: $(TESTNAME)
	 LIBOMPTARGET_INFO=8 ./$(TESTNAME)  2>&1 | $(AOMP)/bin/FileCheck check.txt
