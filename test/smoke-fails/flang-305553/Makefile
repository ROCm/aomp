include ../../Makefile.defs

TESTNAME        = flang-305553
TESTSRC_MAIN    = flang-305553.f90
TESTSRC_AUX     = fortran_callable_init.o
TESTSRC_ALL     = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG           = hipcc
HIP_CLANG_PATH ?= $(AOMPHIP)/bin
HIPCC          ?= $(HIP_CLANG_PATH)/$(CLANG)
# COMPILE_ENV     = HIP_CLANG_PATH=$(HIP_CLANG_PATH)

CFLAGS          = -O3
FLANG           = flang
OMP_BIN         = $(AOMP)/bin/$(FLANG)
CC              = $(OMP_BIN) $(VERBOSE)
EXTRA_CFLAGS    = -L$(AOMPHIP)/lib -lamdhip64 -Wl,-rpath,$(AOMPHIP)/lib

include ../Makefile.rules

all: $(TESTNAME)

# always rebuild
.PHONY: fortran_callable_init.o

fortran_callable_init.o : fortran_callable_init.cpp
	$(COMPILE_ENV) $(HIPCC) -c --offload-arch=$(AOMP_GPU) $^ -o $@
