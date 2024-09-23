include ../../Makefile.defs

TESTNAME     = multi-image
TESTSRC_MAIN = multi-image.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

OMP_FLAGS    = -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$(AOMP_GPU) -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx803
CLANG        = clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
run: $(TESTNAME)
	./$(TESTNAME)
	strings $(TESTNAME) |grep -i gfx
