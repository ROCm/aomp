include ../../Makefile.defs

TESTNAME     = wgs64
TESTSRC_MAIN = wgs64.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang -mllvm  -amdgpu-dump-hsa-metadata 
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
run:
	LIBOMPTARGET_KERNEL_TRACE=1 ./$(TESTNAME)
