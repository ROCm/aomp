include ../../Makefile.defs

TESTNAME     = libgomp-292348
TESTSRC_MAIN = libgomp-292348.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = hipcc
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
OMP_FLAGS    = -lgomp
CFLAGS      = -g -O0 -m64 -fopenmp -lgomp -lstdc++
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
