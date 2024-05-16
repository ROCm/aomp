include ../../Makefile.defs

TESTNAME     = nwchem-s1_1
TESTSRC_MAIN = nwchem-s1_1.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
RUNENV      += LIBOMPTARGET_KERNEL_TRACE=2

FLANG        = flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
run: $(TESTNAME)
	ulimit -s unlimited ; \
	$(AOMP)/bin/rocprof --stats ./$(TESTNAME) > run.log 2>&1 ; \
	python3 printTime.py
