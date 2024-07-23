include ../../Makefile.defs

TESTNAME     = openmp-gpu-burn
TESTSRC_MAIN = openmp-gpu-burn.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
RUNENV      += ROCR_VISIBLE_DEVICES=0
$(info $(AOMP))

BURN         = 0
CLANG        = clang -DBURN=$(BURN)

OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

UNSUPPORTED = ASAN_RUNTIME

include ../Makefile.rules
run: $(TESTNAME)
	$(RUNENV) ./$(TESTNAME)

