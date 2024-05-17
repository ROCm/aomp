include ../../Makefile.defs

TESTNAME     = veccopy
TESTSRC_MAIN = veccopy.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE) -I$(AOMP_REPOS)/llvm-project/openmp/libomptarget/test/ompTest/include -L$(AOMP_REPOS)/build/openmp/libomptarget -lomptest
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
