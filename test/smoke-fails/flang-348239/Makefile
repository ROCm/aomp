include ../../Makefile.defs

TESTNAME     = flang-348239
TESTSRC_AUX  = flang-348239.f
TESTSRC_MAIN =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        = flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)

include ../Makefile.rules
