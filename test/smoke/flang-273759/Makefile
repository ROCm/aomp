include ../../Makefile.defs

TESTNAME     = flang-273759
TESTSRC_MAIN = flang-273759.f95
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

RUNENV       = LIBOMPTARGET_KERNEL_TRACE=1
RUNCMD       = ./$(TESTNAME) 1 2>&1 | $(FILECHECK) $(TESTSRC_MAIN)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules

