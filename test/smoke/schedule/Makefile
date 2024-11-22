include ../../Makefile.defs

TESTNAME     = schedule
TESTSRC_MAIN = schedule.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

UNSUPPORTED = ASAN_RUNTIME

include ../Makefile.rules

run: $(TESTNAME)
	$(RUNENV) $(RUNPROF) ./$(TESTNAME) >$@.log
