include ../Makefile.defs

TESTNAME     = flangtrace
TESTSRC_MAIN = flangtrace.f90
TESTSRC_AUX  = hiptrace.f90
TESTSRC_ALL  = $(TESTSRC_AUX) $(TESTSRC_MAIN)

TARGET       = -fopenmp -O3 -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$(AOMP_GPU) -L/opt/rocm-4.1.0/lib/ -lroctx64 -lroctracer64

FLANG        = flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
run:
	LIBOMPTARGET_KERNEL_TRACE=2 ./$(TESTNAME)
