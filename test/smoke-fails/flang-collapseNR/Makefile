include ../Makefile.defs

ROCM         ?= $(AOMP)/
TESTNAME     = flang-collapseNR
TESTSRC_MAIN = flang-collapseNR.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_AUX) $(TESTSRC_MAIN)

TARGET       = -fopenmp -O3 -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$(AOMP_GPU) 

FLANG        = flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
