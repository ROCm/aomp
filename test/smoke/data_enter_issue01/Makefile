include ../../Makefile.defs

TESTNAME     = data_enter_issue01
TESTSRC_MAIN = data_enter_issue01.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
CFLAGS = -O0

#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
