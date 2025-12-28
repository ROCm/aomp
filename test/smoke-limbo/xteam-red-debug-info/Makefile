include ../../Makefile.defs

TESTNAME     = xteam_red_debug_info
TESTSRC_MAIN = xteam_red_debug_info.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
RUNENV       += LIBOMPTARGET_INFO=1
RUNCMD       = ./$(TESTNAME) 2>&1 | $(FILECHECK) $(TESTSRC_MAIN)

CLANG        ?= clang -g
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
