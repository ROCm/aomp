include ../../Makefile.defs

TESTNAME     = flang_red_swdev-273281-spmd
TESTSRC_MAIN = flang_red_swdev-273281-spmd.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules

run: $(TESTNAME)
	LIBOMPTARGET_KERNEL_TRACE=1 \
	$(RUNENV) ./$(TESTNAME) 1

