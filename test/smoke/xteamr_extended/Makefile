# OVERFLOW_GUARD=1 needs to be set before including Makefile.defs.
# It will ensure that the first argument to the make run command is
# the available GPU memory in GiB. No need to set ARGS variable
# explicitly.
OVERFLOW_GUARD=1

include ../../Makefile.defs

TESTNAME     = test_xteamr
TESTSRC_MAIN = test_xteamr.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        ?= clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)

CFLAGS       = -O3 -std=c++11 -fopenmp-target-fast
ifneq ("$(wildcard $(AOMP)/lib/libmlir_float16_utils.so)","")
  CFLAGS += -lmlir_float16_utils
endif
#-ccc-print-phases
#"-\#\#\#"

UNSUPPORTED = ASAN_COMPILE

include ../Makefile.rules
