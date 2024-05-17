NOOPT        = 1
NOOMP        = 1
include ../../Makefile.defs
OMP_FLAGS    = --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/9

TESTNAME     = hello
TESTSRC_MAIN = hello.f
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang-new
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
