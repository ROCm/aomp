include ../Makefile.defs

TESTNAME     = gdb_teams
TESTSRC_MAIN = gdb_teams.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CFLAGS	     = -g
CLANG        = clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules

run:
	/opt/rocm/bin/rocgdb ./gdb_teams -x cmd_script -q

