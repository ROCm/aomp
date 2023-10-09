include ../../Makefile.defs

TESTNAME     = veccopy-ompt-target-map
TESTSRC_MAIN = veccopy-ompt-target-map.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

RUNCMD       = ./$(TESTNAME) | $(FILECHECK) $(TESTSRC_MAIN)

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE) $(OMPTEST)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
run:
	$(RUNCMD)
