include ../../Makefile.defs

TESTNAME     = clangtrace
TESTSRC_MAIN = clangtrace.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

RUNENV       = LIBOMPTARGET_KERNEL_TRACE=2

CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
