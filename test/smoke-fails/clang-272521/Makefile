include ../../Makefile.defs

TESTNAME     = clang-272521
TESTSRC_MAIN = clang-272521.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang -lm
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
OMP_FLAGS += -O0
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
