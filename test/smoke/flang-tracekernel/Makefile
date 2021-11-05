include ../../Makefile.defs

ROCM         ?= $(AOMP)
TESTNAME     = flangtrace
TESTSRC_MAIN = flangtrace.f90
TESTSRC_AUX  = hiptrace.f90
TESTSRC_ALL  = $(TESTSRC_AUX) $(TESTSRC_MAIN)

TARGET       +=  -Wl,--rpath,$(ROCM)/lib/ -lroctx64 -lroctracer64 -lamdhip64

FLANG        = flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

RUNENV += PATH=$(ROCM)/bin:$(PATH) LIBOMPTARGET_KERNEL_TRACE=2
RUNPROF = rocprof --roctx-trace --sys-trace

include ../Makefile.rules
run:
	$(RUNENV) $(RUNPROF) ./$(TESTNAME)
