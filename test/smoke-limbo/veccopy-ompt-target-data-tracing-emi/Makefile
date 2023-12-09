include ../../Makefile.defs

TESTNAME     = veccopy-ompt-target-data-tracing-emi
TESTSRC_MAIN = veccopy-ompt-target-data-tracing-emi.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

# temporary workaround?
RUNENV      += ROCR_VISIBLE_DEVICES=0

RUNCMD      = ./$(TESTNAME) | $(FILECHECK) $(TESTSRC_MAIN)

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
