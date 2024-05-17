include ../../Makefile.defs

TESTNAME     = host_teams_numa_affinity
TESTSRC_MAIN = host_teams_numa_affinity.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

TARGET       =

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"i

RUNENV += OMP_AFFINITY_FORMAT="Team num: %t, Thread num: %n Affinity: %A" OMP_DISPLAY_AFFINITY=TRUE
RUNENV1 += $(RUNENV) OMP_NUM_TEAMS=2
RUNENV2 += $(RUNENV) OMP_NUM_TEAMS=3
RUNENV3 += $(RUNENV) OMP_NUM_TEAMS=4

include ../Makefile.rules
run:
	$(RUNENV) ./$(TESTNAME)
	$(RUNENV1) ./$(TESTNAME)
	$(RUNENV2) ./$(TESTNAME)
	$(RUNENV3) ./$(TESTNAME)
