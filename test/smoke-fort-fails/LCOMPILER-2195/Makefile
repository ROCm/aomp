include ../../Makefile.defs

TESTNAME     = dc_reduce_test
TESTSRC_MAIN = dc_reduce_test.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
CFLAGS       = -fdo-concurrent-to-openmp=device
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

RUNCMD       = timeout 1m ./${TESTNAME}

include ../Makefile.rules
