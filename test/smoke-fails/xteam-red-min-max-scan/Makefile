include ../../Makefile.defs

TESTNAME     = xteam-red-min-max-scan
TESTSRC_MAIN = xteam-red-min-max-scan.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CFLAGS       += -fopenmp-target-xteam-scan -lm -O0
CLANG        = clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
