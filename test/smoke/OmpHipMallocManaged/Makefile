include ../Makefile.defs

TESTNAME     = OmpHipMallocManaged
TESTSRC_MAIN = OmpHipMallocManaged.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

ITERS ?= -DITERS=1
LARGE ?= -DLARGE=0
USE_MALLOC ?= -DUSE_MALLOC=0
ifeq ($(EPSDB),1)
AOMPHIP ?= $(AOMP)/..
else
AOMPHIP ?= $(AOMP)
endif

VERS = $(shell $(AOMP)/bin/clang --version | grep -oP '(?<=clang version )[0-9.]+')
ifeq ($(shell expr $(VERS) \>= 12.0), 1)
  RPTH = -Wl,-rpath,$(AOMPHIP)/lib
  LLIB = -L$(AOMPHIP)/lib
endif
CLANG        = clang++ $(LLIB) -lamdhip64 $(RPTH) $(USE_MALLOC) $(LARGE) $(ITERS)
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
ifeq ($(EPSDB),1)
EXTRA_LDFLAGS       = -L/opt/rocm/lib
endif
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
RUNENV += LIBOMPTARGET_KERNEL_TRACE=1
