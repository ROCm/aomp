include ../../Makefile.defs

TESTNAME     = test
TESTSRC_MAIN = test.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
TARGET = -O2 -fopenmp -march=znver1 -target x86_64-pc-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908

#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
