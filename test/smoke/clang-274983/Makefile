include ../../Makefile.defs
HSA_XNACK        ?= 1
ifeq ("$(AOMP_GPU)","gfx1250")
  AOMP_TARGET_FEATURES =
else ifeq ("$(AOMP_GPU)","gfx1251")
  AOMP_TARGET_FEATURES =
else
  AOMP_TARGET_FEATURES = :xnack+
endif

TESTNAME     = clang-274983
TESTSRC_MAIN = clang-274983.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

ifeq (sm_,$(findstring sm_,$(AOMP_GPU)))
  _gputriple = nvptx64-nvidia-cuda
else
  _gputriple = amdgcn-amd-amdhsa
endif

SUPPORTED    = $(SUPPORTS_USM)

TARGET       = -fopenmp -O3 -fopenmp-targets=$(_gputriple) -Xopenmp-target=$(_gputriple) -march=$(AOMP_GPU)$(AOMP_TARGET_FEATURES)

CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

RUNENV       += HSA_XNACK=$(HSA_XNACK)
RUNCMD       = ./$(TESTNAME)


include ../Makefile.rules
