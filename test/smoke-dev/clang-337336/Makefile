include ../../Makefile.defs

TESTNAME     = clang-337336
TESTSRC_MAIN = clang-337336.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang++
CFLAGS       = -O0
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

export TIMEOUT = 120s

include ../Makefile.rules
