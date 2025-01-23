include ../../Makefile.defs

TESTNAME     = omp_task_dispatch
TESTSRC_MAIN = omp_task_dispatch.cpp
TESTSRC_AUX  = kernel.cpp
TESTSRC_ALL  = $(TESTSRC_MAIN)

CLANG        = clang++
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)

LINK_FLAGS = -lamdhip64 -L$(AOMP)/../../lib

include ../Makefile.rules

kernel.o: $(TESTSRC_AUX)
	$(CC) -x hip --offload-arch=gfx90a -c -x hip $(TESTSRC_AUX) -o kernel.o

$(TESTNAME): $(TESTSRC_ALL) kernel.o
