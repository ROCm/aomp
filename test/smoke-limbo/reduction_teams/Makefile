include ../../Makefile.defs

TESTNAME     = reduction_teams
TESTSRC_MAIN = reduction_teams.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
RUNENV       = OMP_MAX_TEAMS_DEFAULT=32768 OMP_NUM_TEAMS=4096

#-ccc-print-phases
#"-\#\#\#"

UNSUPPORTED = ASAN_COMPILE

include ../Makefile.rules
