include ../../Makefile.defs

TESTNAME     = check-xnack
TESTSRC_MAIN = check-xnack.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
RUNENV      += HSA_XNACK=1 OMPX_APU_MAPS=1

RUNCMD       = ./$(TESTNAME) 2>&1 | $(FILECHECK) $(TESTSRC_MAIN)

OMP_FLAGS    = -fopenmp --offload-arch=$(AOMP_GPU) -I$(AOMP)/include -I$(AOMP)../../include  -L$(AOMP)/lib -L$(AOMP)/../../lib -lhsa-runtime64
CFLAGS       += -O3
CLANG        = clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)

SUPPORTED    = $(SUPPORTS_USM)

#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
