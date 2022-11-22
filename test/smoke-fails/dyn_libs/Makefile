include ../../Makefile.defs

TESTNAME        = dyn_libs
TESTSRC_MAIN    = main.cpp
TESTSRC_AUX     =
TESTSRC_ALL     = $(TESTSRC_MAIN) $(TESTSRC_AUX)

CLANG           ?= clang++
OMP_BIN         = $(AOMP)/bin/$(CLANG)
CC              = $(OMP_BIN) $(VERBOSE)
EXTRA_LDFLAGS   = -L. -lA -lB
EXTRA_OMP_FLAGS =

libA.so: libA.cpp libA.h Makefile.dynlibs
	AOMP_GPU=$(AOMP_GPU) make -f Makefile.dynlibs libA.so

libB.so: libB.cpp libB.h Makefile.dynlibs
	AOMP_GPU=$(AOMP_GPU) make -f Makefile.dynlibs libB.so

clean ::
	make -f Makefile.dynlibs clean

include ../Makefile.rules

# override default run rule
run:
	HSA_XNACK=1 LD_LIBRARY_PATH=${PWD} ./dyn_libs

