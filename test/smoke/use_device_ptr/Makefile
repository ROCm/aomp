include ../../Makefile.defs

TESTNAME     = use_device_ptr
TESTSRC_MAIN = use_device_ptr.f95
TESTSRC_AUX  = fortran_callable_init.o
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

fortran_callable_init.o : fortran_callable_init.hip
	HIP_CLANG_PATH=$(HIP_CLANG_PATH) $(HIPCC) -c --offload-arch=$(AOMP_GPU) -fPIC $^ -o $@
