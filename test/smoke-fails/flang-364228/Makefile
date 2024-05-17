include ../../Makefile.defs

TESTNAME     = flang-364228
TESTSRC_MAIN = flang-364228.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        = flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
OMP_FLAGS   += -DSHOW_BUG -DUSE_MATMUL -DUSE_GPU -USIMD

include ../Makefile.rules
