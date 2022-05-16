include ../../Makefile.defs

TESTNAME     = clang-293471
TESTSRC_MAIN = clang-293471.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

OMP_FLAGS = -fnt-store=aggressive -O3 -march=znver2 -S
CLANG        = clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
run:
	cat clang-293471.s
