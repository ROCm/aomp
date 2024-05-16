include ../../Makefile.defs

TESTNAME     = flang-305553
TESTSRC_MAIN = flang-305553.f90
TESTSRC_AUX  = fortran_callable_init.o
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
AOMPHIP ?= $(AOMP)
HIPCC ?= $(AOMPHIP)/bin/hipcc

CFLAGS = -O3
FLANG        = flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
EXTRA_CFLAGS = -L$(AOMPHIP)/lib -lamdhip64 -Wl,-rpath,$(AOMPHIP)/lib
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
all: $(TESTNAME)

fortran_callable_init.o : fortran_callable_init.cpp
	$(HIPCC) -c --offload-arch=$(AOMP_GPU) $^ -o $@
