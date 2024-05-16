include ../../Makefile.defs

TESTNAME     = libgomp-292348
TESTSRC_MAIN = libgomp-292348.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= hipcc
OMP_FLAGS    = -lgomp
CFLAGS      = -g -O0 -m64 -fopenmp -lgomp -lstdc++

ifeq ($(EPSDB),1)
NEWEST_ROCM ?= $(shell ls --sort=time /opt | grep -m 1 rocm)
ifeq ($(NEWEST_ROCM),)
$(error "Error: No ROCm detected!")
endif
AOMPROCM ?=/opt/$(NEWEST_ROCM)
$(warning "AOMPROCM = $(AOMPROCM)")
OMP_BIN      = $(AOMPROCM)/bin/$(CLANG)
else
AOMP ?=$(AOMPROCM)
OMP_BIN      = $(AOMP)/bin/$(CLANG)
endif
HIP_CLANG_PATH ?= $(AOMP)/bin
CC           = HIP_CLANG_PATH=$(HIP_CLANG_PATH) $(OMP_BIN) $(VERBOSE)

#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
