include ../../Makefile.defs

TESTNAME     = clang-gdb1
TESTSRC_MAIN = clang-gdb1.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang
CFLAGS		 = -ggdb -O0 -mllvm -amdgpu-spill-cfi-saved-regs -gheterogeneous-dwarf -g -lm 
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
