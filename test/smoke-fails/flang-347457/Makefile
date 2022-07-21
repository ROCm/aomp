include ../../Makefile.defs

TESTNAME     = flang-347457
TESTSRC_AUX  = flang-347457.f90
TESTSRC_MAIN =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        = flang -DUSE_GPU
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)

include ../Makefile.rules
