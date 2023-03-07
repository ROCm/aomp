include ../../Makefile.defs

TESTNAME     = no-offload-arch
TESTSRC_MAIN = no-offload-arch.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
CFLAGS =    -O2    -fopenmp --no-offload-arch=gfx90a
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
