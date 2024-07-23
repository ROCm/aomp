include ../../Makefile.defs

TESTNAME     = MapWritesIntoConstGlobalStruct
TESTSRC_MAIN = map-from-into-const-global-struct.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CFLAGS       += -O3 -fno-openmp-target-ignore-env-vars
CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules

run:
	OMPX_FORCE_SYNC_REGIONS=0 ./$(TESTNAME)
	OMPX_FORCE_SYNC_REGIONS=1 ./$(TESTNAME)

