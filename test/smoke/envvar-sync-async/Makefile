include ../../Makefile.defs

TESTNAME     = EnvVar-SyncAsync
TESTSRC_MAIN = envvar-sync-async.c
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
	echo "Running async (default)"
	./$(TESTNAME)
	echo ""
	echo "Running sync"
	OMPX_FORCE_SYNC_REGIONS=1 ./$(TESTNAME)
	echo ""
	echo "Runnincg forced async"
	OMPX_FORCE_SYNC_REGIONS=0 ./$(TESTNAME)
