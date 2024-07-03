include ../../Makefile.defs

TESTNAME     = flang-386800
TESTSRC_MAIN = flang-386800.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
OMP_FLAGS    = -g
CFLAGS       = -O0
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

UNSUPPORTED = ASAN_COMPILE

include ../Makefile.rules
run:
	./$(TESTNAME)
	$(AOMP)/bin/llvm-dwarfdump -v $(TESTNAME) 2>&1 | tee out
	grep -L DW_AT_lower_bound out
