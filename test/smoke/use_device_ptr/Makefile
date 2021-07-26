include ../../Makefile.defs

TESTNAME     = use_device_ptr
TESTSRC_MAIN = use_device_ptr.f95
TESTSRC_AUX  = fortran_callable_init.o
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CFLAGS = -O3
FLANG        = flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
EXTRA_CFLAGS = -lamdhip64
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
all: $(TESTNAME)

fortran_callable_init.o : fortran_callable_init.hip
	$(AOMP)/bin/hipcc -c --offload-arch=$(AOMP_GPU) $^ -o $@
