include ../../Makefile.defs

TESTNAME       = vmulsum-hsa-stats
TESTSRC_MAIN   = main.c
TESTSRC_AUX    = vmul.c vsum.c
TESTSRC_ALL    = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG          ?= clang
OMP_BIN        = $(AOMP)/bin/$(CLANG)
CC             = $(OMP_BIN) $(VERBOSE)
RUNPROF = $(AOMPHIP)/bin/rocprofv3 --output-format csv --hsa-trace --stats -- 

#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules

