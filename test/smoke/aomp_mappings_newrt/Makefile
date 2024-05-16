include ../../Makefile.defs

TESTNAME     = aomp_mappings_newrt
TESTSRC_MAIN = aomp_mappings_newrt.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

RUNENV       = LIBOMPTARGET_AMDGPU_SMALL_BLOCKSIZE=256

CLANG        = clang -O2 -mllvm -openmp-opt-disable=true
OMP_BIN      = $(AOMP)/bin/$(CLANG) -isystem$(AOMP)/include
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
