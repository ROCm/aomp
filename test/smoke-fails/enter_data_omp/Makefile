include ../../Makefile.defs

TESTNAME     = enter_data_omp
TESTSRC_MAIN = enter_data_omp.cpp
TESTSRC_ALL  = $(TESTSRC_MAIN)
HIP_INC_DIR  = $(AOMP)/include
HSA_XNACK    ?= 1
RUNENV       += HSA_XNACK=$(HSA_XNACK) LIBOMPTARGET_KERNEL_TRACE=3
SUPPORTED    = $(SUPPORTS_USM)
RUNCMD       = ./$(TESTNAME)

CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
