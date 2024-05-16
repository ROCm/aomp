include ../../Makefile.defs

TESTNAME     = stream
TESTSRC_MAIN = main.cpp
TESTSRC_AUX  = OMP45Stream.cpp
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
CFLAGS       = -DOMP45 -std=c++11  -fopenmp-gpu-threads-per-team=1024 -fopenmp-target-fast -O3
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
