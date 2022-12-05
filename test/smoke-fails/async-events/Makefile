include ../../Makefile.defs

TESTNAME     = AsyncEvents
TESTSRC_MAIN = async-events.cpp
TESTSRC_AUX  = empty-sink.c
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
run:
	echo $(PATH)
	./$(TESTNAME)
