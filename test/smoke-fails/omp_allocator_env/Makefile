include ../../Makefile.defs

TESTNAME     = omp_allocator_env
TESTSRC_MAIN = omp_allocator_env.cpp
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

ALIGN_VAL    = 4096
CLANG        = clang++ -DALIGN_VAL=$(ALIGN_VAL)
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules

run:
	OMP_ALLOCATOR=omp_default_mem_alloc:alignment=$(ALIGN_VAL) ./omp_allocator_env
