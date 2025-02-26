include ../../Makefile.defs

TESTNAME     = xteam-red-min-max
TESTSRC_MAIN = xteam-red-min-max.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

EXTRA_LDFLAGS     += -lm
CLANG        = clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
