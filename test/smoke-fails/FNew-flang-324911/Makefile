include ../../Makefile.defs

TESTNAME     = flang-324911
TESTSRC_MAIN = flang-324911.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
CFLAGS       = -g -O0
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)

include ../Makefile.rules
run: $(TESTNAME)
	$(RUNENV) ./$(TESTNAME)
	readelf --debug-dump=rawline $(TESTNAME) | grep  "Raw dump of debug"
