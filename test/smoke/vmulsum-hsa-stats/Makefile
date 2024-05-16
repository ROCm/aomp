include ../../Makefile.defs

TESTNAME       = vmulsum-hsa-stats
TESTSRC_MAIN   = main.c
TESTSRC_AUX    = vmul.c vsum.c
TESTSRC_ALL    = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG          ?= clang
OMP_BIN        = $(AOMP)/bin/$(CLANG)
CC             = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
run:
	$(AOMP)/bin/rocprof --hsa-trace ./$(TESTNAME)
