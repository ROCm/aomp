include ../../Makefile.defs

TESTNAME     = aomp_mappings_newrt
TESTSRC_MAIN = aomp_mappings_newrt.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang -fopenmp-target-new-runtime
OMP_BIN      = $(AOMP)/bin/$(CLANG) -isystem$(AOMP)/include
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
