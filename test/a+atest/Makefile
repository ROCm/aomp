include ../Makefile.defs

TESTNAME     = stream_omp_a+a
TESTSRC_MAIN = stream.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

OMP_FLAGS = -O3 -mcmodel=large -DSTREAM_TYPE=double -mavx2  -DSTREAM_ARRAY_SIZE=100000 -DNTIMES=10 -ffp-contract=fast -march=znver2 -fopenmp -fnt-store

CLANG        = clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../smoke/Makefile.rules

run:
	export OMP_SCHEDULE=static ; \
	export OMP_DYNAMIC=false ; \
	export OMP_STACKSIZE=256M ; \
	export GOMP_CPU_AFFINITY=0-32:8 ; \
	export OMP_NUM_THREADS=16 ; \
	./$(TESTNAME)


