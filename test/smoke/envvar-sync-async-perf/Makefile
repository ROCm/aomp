include ../../Makefile.defs

TESTNAME     = Target-Map-ToFrom-Performance
TESTSRC_MAIN = target-map.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CFLAGS       += -O3 -fno-openmp-target-ignore-env-vars
CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules

run:
	echo "Running async (default)"
	OMPX_FORCE_SYNC_REGIONS=0 ./$(TESTNAME)
	echo "Running sync"
	OMPX_FORCE_SYNC_REGIONS=1 ./$(TESTNAME)
