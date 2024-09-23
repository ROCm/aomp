include ../../Makefile.defs

TESTNAME     = many_arrays
TESTSRC_MAIN = many_arrays.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"



ifeq ($(ISVIRT),1)
ARGS = $(GPU_MEM_SIZE_GIB)
endif

RUNENV = LIBOMPTARGET_KERNEL_TRACE=2

export TIMEOUT = 120s

include ../Makefile.rules

