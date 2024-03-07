include ../../Makefile.defs

TESTNAME     = map-always
TESTSRC_MAIN = map-always.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CFLAGS       += -O3
CLANG        = clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)

SUPPORTED    = gfx90a,gfx940,gfx941,gfx942

#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
