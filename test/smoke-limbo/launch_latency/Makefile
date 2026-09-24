include ../../Makefile.defs

TESTNAME     = launch_latency
TESTSRC_MAIN = launch_latency.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
RUNENV       = ulimit -s unlimited;
RUNCMD       = bash ./run_launch_latency.sh $(AOMPHIP)

#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
