include ../../Makefile.defs

TESTNAME     = wgs64
TESTSRC_MAIN = wgs64.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
RUNENV      += LIBOMPTARGET_KERNEL_TRACE=1

CLANG        ?= clang -mllvm  -amdgpu-dump-hsa-metadata -save-temps
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
run:
	./$(TESTNAME)
	fgrep -s '.max_flat_workgroup_size: 64' wgs64-openmp-amdgcn-amd-amdhsa-*.s

