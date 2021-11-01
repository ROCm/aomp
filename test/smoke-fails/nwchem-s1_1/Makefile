include ../../Makefile.defs

TESTNAME     = nwchem-s1_1
TESTSRC_MAIN = nwchem-s1_1.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)


FLANG        = flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
run: $(TESTNAME)
	ulimit -s unlimited ;\
	./$(TESTNAME)
