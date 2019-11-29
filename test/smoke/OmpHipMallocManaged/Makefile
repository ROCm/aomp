include ../Makefile.defs

TESTNAME     = OmpHipMallocManaged
TESTSRC_MAIN = OmpHipMallocManaged.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

ifeq ($(LARGE),1)
  LARGE = -DLARGE
endif
ifeq ($(USE_MALLOC),1)
  USE_MALLOC = -DUSE_MALLOC
endif

CLANG        = clang++ -lhip_hcc $(USE_MALLOC) $(LARGE)
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
RUNENV += LIBOMPTARGET_KERNEL_TRACE=1
