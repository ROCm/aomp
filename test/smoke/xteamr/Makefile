include ../../Makefile.defs

TESTNAME     = xteamr
TESTSRC_MAIN = xteamr.cpp
TESTSRC_AUX  = 
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
CFLAGS       = -O3 -std=c++11 -latomic
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
