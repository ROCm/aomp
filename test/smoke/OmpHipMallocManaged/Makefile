include ../Makefile.defs

TESTNAME     = OmpHipMallocManaged
TESTSRC_MAIN = OmpHipMallocManaged.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

ITERS ?= -DITERS=1
LARGE ?= -DLARGE=0
USE_MALLOC ?= -DUSE_MALLOC=0

CLANG        = clang++ -lhip_hcc $(USE_MALLOC) $(LARGE) $(ITERS)
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
RUNENV += LIBOMPTARGET_KERNEL_TRACE=1
