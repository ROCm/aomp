include ../../Makefile.defs

TESTNAME     = metadirective
TESTSRC_MAIN = metadirective.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang 
OMP_BIN      = $(AOMP)/bin/$(CLANG) -fopenmp-gpu-threads-per-team=257
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
