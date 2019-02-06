include ../Makefile.defs

TESTNAME      = issue_apps_001
TESTSRC_MAIN  = lulesh.cc
TESTSRC_AUX   =
TESTSRC_ALL   = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG         = clang
OMP_BIN       = $(AOMP)/bin/$(CLANG)
CC            = $(OMP_BIN) $(VERBOSE)
EXTRA_CFLAGS  = -DUSE_MPI=0
EXTRA_LDFLAGS = -lm

include ../Makefile.rules
