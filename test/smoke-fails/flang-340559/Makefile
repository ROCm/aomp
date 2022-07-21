include ../../Makefile.defs

TESTNAME     = flang-340559
TESTSRC_AUX  = main.f90
TESTSRC_MAIN = matrix.f90
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        = flang -DUSE_GPU
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)

include ../Makefile.rules
