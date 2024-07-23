include ../../Makefile.defs

TESTNAME     = clang-jmc-dev
TESTSRC_MAIN = clang-jmc-dev.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
RUNENV      += ROCR_VISIBLE_DEVICES=0
$(info $(AOMP))

CLANG        = clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"


UNSUPPORTED = ASAN_RUNTIME

include ../Makefile.rules
run: $(TESTNAME)
	$(RUNENV) ./$(TESTNAME)

