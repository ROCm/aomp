include ../../Makefile.defs

TESTNAME     = clang-274983
TESTSRC_MAIN = clang-274983.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

ifeq (sm_,$(findstring sm_,$(AOMP_GPU)))
  _gputriple = nvptx64-nvidia-cuda
else
  _gputriple = amdgcn-amd-amdhsa
endif

TARGET       = -fopenmp -O3 -fopenmp-targets=$(_gputriple) -Xopenmp-target=$(_gputriple) -march=$(AOMP_GPU)

CLANG        = clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
