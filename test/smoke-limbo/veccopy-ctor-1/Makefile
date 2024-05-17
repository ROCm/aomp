include ../../Makefile.defs

TESTNAME     = veccopy-ctor-1
TESTSRC_MAIN = veccopy.cpp
TESTSRC_AUX  = veccopy-ctor-1.cpp
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules

