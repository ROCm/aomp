include ../../Makefile.defs

TESTNAME     = clang-nwchem-s1_1
TESTSRC_MAIN = clang-nwchem-s1_1.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
RUNENV      += LIBOMPTARGET_KERNEL_TRACE=2

CLANG        = clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
run: $(TESTNAME)
	ulimit -s unlimited ;\
	$(RUNENV) ./$(TESTNAME) > run.log 2>&1 ; \
	grep target_team_region run.log | awk '{print $$2, $$3}' 

