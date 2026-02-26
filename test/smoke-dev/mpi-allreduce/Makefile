include ../../Makefile.defs
TESTNAME   = mpi-allreduce
TESTSRC    = mpi-allreduce.cpp
AOMP      ?= $(HOME)/rocm/aomp
CFLAGS     = -O3 -fopenmp --offload-arch=$(AOMP_GPU)
AOMP_SUPP ?= $(HOME)/local
MPI       ?= $(AOMP_SUPP)/openmpi
CC_MPI     = $(MPI)/bin/mpic++
CLANG_DIR  = $(AOMP)/lib/llvm/bin
PATH      += $(CLANG_DIR)

$(TESTNAME): $(TESTSRC)
	OMPI_CC=$(CLANG_DIR)/clang++ $(CC_MPI) $(CFLAGS) $(LFLAGS) $^ -o $@

run: $(TESTNAME)
	$(MPI)/bin/mpirun -np 2  $(AOMP)/bin/gpurun -v ./$(TESTNAME)

clean:
	rm -f $(TESTNAME) obin *.i *.ii *.bc *.lk a.out-* *.ll *.s *.o *.cubin
