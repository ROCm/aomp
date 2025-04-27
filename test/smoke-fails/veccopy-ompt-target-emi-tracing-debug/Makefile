include ../../Makefile.defs

TESTNAME     = veccopy-ompt-target-emi-tracing
TESTSRC_MAIN = veccopy-ompt-target-emi-tracing.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

RUNCMD      = ./$(TESTNAME) | $(FILECHECK) $(TESTSRC_MAIN)

CFLAGS     += -fopenmp-runtimelib=lib-debug
CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
