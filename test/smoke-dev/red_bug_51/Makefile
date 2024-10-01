include ../../Makefile.defs

TESTNAME     = red_bug_51
TESTSRC_MAIN = red_bug_51.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
CFLAGS       = -O0

#-ccc-print-phases
#"-\#\#\#"

UNSUPPORTED = ASAN_RUNTIME

include ../Makefile.rules
