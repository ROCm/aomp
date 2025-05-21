include ../../Makefile.defs

TESTNAME     = flang-host-targ
TESTSRC_MAIN = flang-host-targ.f90

TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

TARGET       = -fopenmp -O3 
TARGET       += -fopenmp-targets=x86_64-pc-linux-gnu -Xopenmp-target=x86_64-pc-linux-gnu --march=znver1

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

UNSUPPORTED = ASAN_COMPILE

include ../Makefile.rules
