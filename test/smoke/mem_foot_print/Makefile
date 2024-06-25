include ../../Makefile.defs

TESTNAME     = mem_foot_print
TESTSRC_MAIN = mem_foot_print.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CFLAGS	     = -save-temps
CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

UNSUPPORTED = ASAN_RUNTIME
include ../Makefile.rules

run:
	readelf -s a.out-openmp-amdgcn-amd-amdhsa-* |grep omptarget_nvptx_device | head -1 

