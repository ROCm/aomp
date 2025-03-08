TESTNAME   = mpi-simple
TESTSRC    = mpi-simple.cpp
AOMP      ?= $(HOME)/rocm/aomp
_gpu       = $(shell $(AOMP)/../../bin/rocm_agent_enumerator | grep -m 1 -E gfx[^0]{1}.{2})
CFLAGS     = -O3 -fopenmp --offload-arch=$(_gpu)
AOMP_SUPP ?= $(HOME)/local
MPI       ?= $(AOMP_SUPP)/openmpi
CC_MPI     = $(MPI)/bin/mpic++
CLANG_DIR  = $(AOMP)/bin
PATH      += $(CLANG_DIR)

$(TESTNAME): $(TESTSRC)
	OMPI_CC=$(CLANG_DIR)/clang++ $(CC_MPI) $(CFLAGS) $(LFLAGS) $^ -o $@

run: $(TESTNAME)
	$(MPI)/bin/mpirun -np 2  $(CLANG_DIR)/gpurun ./$(TESTNAME)

clean:
	rm -f $(TESTNAME) obin *.i *.ii *.bc *.lk a.out-* *.ll *.s *.o *.cubin
