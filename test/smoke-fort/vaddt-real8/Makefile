include ../../Makefile.defs

TESTNAME     = vaddt
TESTSRC_MAIN = vaddt.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
CFLAGS       = -cpp -DCHKTYPE='real(8)' -DCHKBYTES=8
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
