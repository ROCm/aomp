include ../../Makefile.defs

TESTNAME     = gdb_teams
TESTSRC_MAIN = gdb_teams.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CFLAGS	     = -ggdb
CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules

AOMPROCM ?= $(AOMP)
ifneq (,$(wildcard $(AOMPROCM)/bin/rocgdb))
RUNPROF   = $(AOMPROCM)/bin/rocgdb
else ifneq (,$(wildcard /opt/rocm/bin/rocgdb))
RUNPROF   = /opt/rocm/bin/rocgdb
else
RUNPROF   = $(AOMPROCM)/../bin/rocgdb
endif
ARGS      = -x cmd_script -q

run:
	$(RUNPROF) $(TESTNAME) $(ARGS)

