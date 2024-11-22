include ../../Makefile.defs

TESTNAME        = vectoradd_hip
TESTSRC_MAIN    = vectoradd_hip.o
TESTSRC_AUX     = vectoradd_hip2.o
TESTSRC_ALL     = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG           = hipcc
AOMPHIP        ?= $(AOMP)
HIP_CLANG_PATH ?= $(AOMPHIP)/bin
HIPCC          ?= $(HIP_CLANG_PATH)/$(CLANG)
COMPILE_ENV     = HIP_CLANG_PATH=$(HIP_CLANG_PATH)

OMP_FLAGS       = -fgpu-rdc
OMP_BIN         = $(HIPCC)
CC              = $(COMPILE_ENV) $(OMP_BIN)

include ../Makefile.rules

$(TESTRC_MAIN)  : vectoradd_hip.cpp
$(TESTSRC_AUX)  : vectoradd_hip2.cpp
