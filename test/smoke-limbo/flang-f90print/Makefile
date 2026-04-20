include ../../Makefile.defs

TESTNAME     = flang-f90print
TESTSRC_MAIN = flang-f90print.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang 
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

RUNCMD       = ./$(TESTNAME) 2>&1 | grep -q 'f90print inside target region' && echo 'Success' || (echo 'Failure' && false)

UNSUPPORTED = ASAN_COMPILE

include ../Makefile.rules
run: $(TESTNAME)
	$(RUNCMD)
