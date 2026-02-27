# OVERFLOW_GUARD=1 needs to be set before including Makefile.defs.
# It will ensure that the first argument to the make run command is
# the available GPU memory in GiB. No need to set ARGS variable
# explicitly.
OVERFLOW_GUARD=1

include ../../Makefile.defs

TESTNAME     = many_arrays
TESTSRC_MAIN = many_arrays.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

RUNENV = LIBOMPTARGET_KERNEL_TRACE=2

export TIMEOUT = 120s

include ../Makefile.rules

