include ../../Makefile.defs

TESTNAME     = nwchem-manually-collapsed-s1_1
TESTSRC_MAIN = nwchem-manually-collapsed-s1_1.f90
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
	ulimit -s unlimited ;\
	$(RUNENV) ./$(TESTNAME) > run.log 2>&1 ; \
	grep target_team_region run.log | awk '{print $$2, $$3}' 

