include ../../Makefile.defs

TESTNAME        = clang-323972
TESTSRC_MAIN    = clang-323972.cpp
TESTSRC_AUX     =
TESTSRC_ALL     = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG           = hipcc
AOMPHIP        ?= $(AOMP)
HIP_CLANG_PATH ?= $(AOMPHIP)/bin
HIPCC          ?= $(HIP_CLANG_PATH)/$(CLANG)
COMPILE_ENV     = HIP_CLANG_PATH=$(HIP_CLANG_PATH)

OMP_FLAGS       = -fopenmp -O2
OMP_BIN         = $(HIPCC)
CC              = $(COMPILE_ENV) $(OMP_BIN) $(VERBOSE)

include ../Makefile.rules
