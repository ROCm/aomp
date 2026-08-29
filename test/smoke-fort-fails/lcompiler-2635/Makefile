include ../../Makefile.defs

TESTNAME     = test
TESTSRC_MAIN = test.f90
TESTSRC_AUX  =
TESTSRC_ALL  =  $(TESTSRC_AUX) $(TESTSRC_MAIN)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
CFLAGS       = $(FLANG_GPU_LINK_FLAGS)
# CFLAGS       = $(FLANG_GPU_LINK_FLAGS) -O0    # works at -O0
#OMP_FLAGS   += -DFAIL

RUNCMD       = ./${TESTNAME} 2>&1 | tee ${TESTNAME}.out 2>&1 && diff -w ${TESTNAME}.out chk.stdout

include ../Makefile.rules
