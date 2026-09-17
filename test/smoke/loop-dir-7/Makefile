include ../../Makefile.defs

TESTNAME     = loop_dir_7
TESTSRC_MAIN = loop_dir_7.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CFLAGS       += -fopenmp-assume-teams-oversubscription -fopenmp-assume-threads-oversubscription -fopenmp-assume-no-nested-parallelism -fopenmp-assume-no-thread-state
CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
