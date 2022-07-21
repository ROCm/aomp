include ../../Makefile.defs

TESTNAME     = clang-host-targ2
TESTSRC_MAIN = clang-host-targ2.o

TESTSRC_AUX  = another.o

$(TESTRC_MAIN): clang-host-targ2.c
$(TESTSRC_AUX): another.c

TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

TARGET       = -fopenmp -O3 
TARGET       += -fopenmp-targets=x86_64-pc-linux-gnu -Xopenmp-target=x86_64-pc-linux-gnu --march=x86

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
