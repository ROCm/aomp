include ../../Makefile.defs

TESTNAME     = kernel-exec-modes-host
TESTSRC_MAIN = kernel-exec-modes-host.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
RUNENV      += OMP_TARGET_OFFLOAD=DISABLED

FLANG        ?= flang-new
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
