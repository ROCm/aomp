include ../../Makefile.defs

TESTNAME     = targetid-multi-image
TESTSRC_MAIN = targetid-multi-image.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

OMP_FLAGS    = -fopenmp --offload-arch=$(AOMP_GPU):xnack+,$(AOMP_GPU):xnack-
CLANG        = clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)

SUPPORTED  = gfx90a,gfx90c,gfx940,gfx941,gfx942,gfx1010,gfx1011,gfx1012,gfx1013

HSA_XNACK   ?= 1
RUNENV       = HSA_XNACK=$(HSA_XNACK)
RUNCMD       = ./$(TESTNAME) && HSA_XNACK=0 ./$(TESTNAME) && strings $(TESTNAME) |grep -i gfx

#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
