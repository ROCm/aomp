include ../../Makefile.defs

TESTNAME     = flang-283928-collapse7
TESTSRC_MAIN = flang-283928-collapse7.f90
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
