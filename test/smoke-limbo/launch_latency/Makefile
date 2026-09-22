include ../../Makefile.defs

TESTNAME     = launch_latency
TESTSRC_MAIN = launch_latency.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
RUNENV       = ulimit -s unlimited;
RUNCMD       = ( $(AOMPHIP)/bin/rocprofv3 --output-format csv --kernel-trace --stats -- ./$(TESTNAME) 2> ./rocprofv3.log
RUNCMD      += && python3 printLatency.py || { status=$$?; cat ./rocprofv3.log; exit $$status; } )

#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
