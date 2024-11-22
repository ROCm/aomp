include ../../Makefile.defs

TESTNAME     = issue_flang_libomp
TESTSRC_MAIN = issue_flang_libomp.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

TARGET       = -fopenmp -O3 -w
FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

UNSUPPORTED = ASAN_COMPILE

include ../Makefile.rules
all: $(TESTNAME)
	@ldd issue_flang_libomp | grep  libomp.so
