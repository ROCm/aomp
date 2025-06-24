include ../../Makefile.defs

TESTNAME     = vmm
TESTSRC_MAIN = vmm.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
CFLAGS       =
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

RUNENV       = LIBOMPTARGET_KERNEL_TRACE=1

include ../Makefile.rules
