include ../../Makefile.defs

TESTNAME     = redprob
TESTSRC_MAIN = redprob.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
run: $(TESTNAME)
	ulimit -s unlimited ; \
	rocprofv3 --output-format csv --kernel-trace --stats -- ./$(TESTNAME) > run.log 2>&1 ; \
	python3 flang-494405.py
