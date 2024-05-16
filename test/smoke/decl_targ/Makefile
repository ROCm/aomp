include ../../Makefile.defs

TESTNAME     = decl_targ
TESTSRC_MAIN = decl_targ.f95
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CFLAGS = -O3
EXTRA_CFLAGS += -w -i8
FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
all: $(TESTNAME)
