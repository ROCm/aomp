include ../../Makefile.defs

TESTNAME     = flang-339704
TESTSRC_MAIN = flang-339704.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        = flang -DUSE_GPU
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)

include ../Makefile.rules
