include ../../Makefile.defs

TESTNAME     = targetid-multi-image
TESTSRC_MAIN = targetid-multi-image.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

OMP_FLAGS    = -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$(AOMP_GPU):xnack- -Xopenmp-target=amdgcn-amd-amdhsa -march=$(AOMP_GPU):xnack+
CLANG        = clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)

UNSUPPORTED  = gfx1030,gfx1031,gfx1032,gfx1033,gfx1034,gfx1035,gfx1036,gfx1100,gfx1101,gfx1102,gfx1103,gfx1150,gfx1151

HSA_XNACK ?= 1
RUNCMD       = HSA_XNACK=$(HSA_XNACK) ./$(TESTNAME) && HSA_XNACK=0 ./$(TESTNAME) && strings $(TESTNAME) |grep -i gfx

#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
