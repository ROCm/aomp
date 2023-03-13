# This test is about the --offload-arch parameter (will be set here).
# The test will either PASS if --no-offload-arch=gfx... (considered 'valid')
# is suppressed or the expected behavior is maintained.
# Prevent corresponding intervention from the Makefile.defs via OMP_HOST.
OMP_HOST=1
include ../../Makefile.defs

TESTNAME     = clang-388240
TESTSRC_MAIN = clang-388240.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang
CFLAGS       = -O3 -fopenmp --offload-arch=$(AOMP_GPU)
CFLAGS      += --no-offload-arch=gfx...
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)

include ../Makefile.rules
