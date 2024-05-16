include ../../Makefile.defs

TESTNAME     = d2h_slow_copy
TESTSRC_MAIN = d2h_slow_copy.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE) -lm -lstdc++
#-ccc-print-phases
#"-\#\#\#"
export TIMEOUT = 120s

include ../Makefile.rules
