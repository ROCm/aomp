include ../../Makefile.defs

TESTNAME     = flang-347457
TESTSRC_AUX  = flang-347457.f90
TESTSRC_MAIN =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
CFLAGS       = -DUSE_GPU -Mx,183,0x400000
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)

include ../Makefile.rules
