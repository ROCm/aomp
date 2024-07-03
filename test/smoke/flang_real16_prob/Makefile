include ../../Makefile.defs

TESTNAME     = flang_real16_prob
TESTSRC_MAIN = flang_real16_prob.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

TARGET       = -lz
OMP_FLAGS    =

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

UNSUPPORTED = ASAN_COMPILE

include ../Makefile.rules
