include ../../Makefile.defs

TESTNAME     = flang-317868
TESTSRC_MAIN = flang-317868.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        = flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
OMP_FLAGS   += -E

include ../Makefile.rules
