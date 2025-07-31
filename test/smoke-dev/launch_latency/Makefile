include ../../Makefile.defs

TESTNAME     = launch_latency
TESTSRC_MAIN = launch_latency.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
RUNENV       = ulimit -s unlimited;
# RUNPROF      = /usr/bin/rocprofv3
# RUNPROF_FLAGS= --output-format csv --kernel-trace --stats -- 
# RUNCMD       = ./$(TESTNAME) 
RUNCMD       = rocprofv3 --output-format csv --kernel-trace --stats -- ./$(TESTNAME) && python3 printLatency.py

#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
