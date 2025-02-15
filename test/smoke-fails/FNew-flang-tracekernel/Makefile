include ../../Makefile.defs

ROCM         ?= $(AOMPHIP)
TESTNAME     = flangtrace
TESTSRC_MAIN = flangtrace.f90
TESTSRC_AUX  = hiptrace.f90
TESTSRC_ALL  = $(TESTSRC_AUX) $(TESTSRC_MAIN)

TARGET       += -Wl,--rpath,$(ROCM)/lib/ -lroctx64 -lroctracer64 -lamdhip64 -L$(ROCM)/lib

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

LIBPATH = $(AOMPHIP)/lib:/opt/rocm/lib
RUNENV += PATH=$(ROCM)/bin:$(PATH) LD_LIBRARY_PATH=$(LIBPATH):$(LD_LIBRARY_PATH) LIBOMPTARGET_KERNEL_TRACE=2
RUNPROF = $(AOMPHIP)/bin/rocprof
RUNPROF_FLAGS = --roctx-trace --sys-trace

UNSUPPORTED = ASAN_COMPILE

include ../Makefile.rules
