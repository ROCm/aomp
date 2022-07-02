include ../../Makefile.defs

TESTNAME     = flang-325322
TESTSRC_MAIN = flang-325322.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
OMP_FLAGS   += -Mstandard -Mstandard -DUSE_GPU -UUSE_MATMUL  \
            -Mfixed -Wall -Mstandard
include ../Makefile.rules
