include ../../Makefile.defs

TESTNAME     = clang-440121
TESTSRC_MAIN = clang-440121.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG        = clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
WRONG_LIB    = `pwd`/lib
CC           = LIBRARY_PATH=$(WRONG_LIB):$(AOMP)/lib $(OMP_BIN) $(VERBOSE)
OMP_FLAGS 	 += --offload-arch=gfx90a -fopenmp

#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
