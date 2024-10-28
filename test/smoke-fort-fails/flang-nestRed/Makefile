include ../../Makefile.defs

TESTNAME     = redprob
TESTSRC_MAIN = redprob.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang-new
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
run: $(TESTNAME)
	ulimit -s unlimited ; \
	rocprof --stats ./$(TESTNAME) > run.log 2>&1 ; \
	column -t -s  ,  < results.stats.csv
