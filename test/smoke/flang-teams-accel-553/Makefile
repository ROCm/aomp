include ../../Makefile.defs

TESTNAME     = flang-teams
TESTSRC_MAIN = flang-teams.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang  -O3 -Mx,234,32
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"
RUNENV      += LIBOMPTARGET_KERNEL_TRACE=1

include ../Makefile.rules
run: $(TESTNAME)
	        $(RUNENV) ./$(TESTNAME) 

