include ../../Makefile.defs

TESTNAME     = targetid
TESTSRC_MAIN = targetid.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

OMP_FLAGS    = -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a:sramecc-:xnack+
CLANG        = clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)

HSA_XNACK ?= 1

#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
run: $(TESTNAME)
	HSA_XNACK=${HSA_XNACK} ./$(TESTNAME)
# 	will pass only on machines where xnack+ is configured

	strings $(TESTNAME) |grep -i gfx
