include ../../Makefile.defs

TESTNAME     = flang-322945
TESTSRC_MAIN = flang-322945.f90

TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
CFLAGS       = -O3

include ../Makefile.rules

no_mpi: $(TESTSRC_MAIN)
	$(OMP_BIN) $(CFLAGS) $(TESTSRC_MAIN) -o $(TESTNAME)
