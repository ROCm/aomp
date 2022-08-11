include ../../Makefile.defs

TESTNAME     = xteam_reductions
TESTSRC_MAIN = xteam_reductions.cpp
TESTSRC_AUX  = 
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
CFLAGS       = -O3 -std=c++11 
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
