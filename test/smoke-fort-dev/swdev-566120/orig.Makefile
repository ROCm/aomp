COMP=amd

ifeq ($(COMP),amd)
FC=amdflang
FCFLAGS=-g -O2 -fopenmp -fPIC --offload-arch=gfx90a,gfx942 -DUSE_ROCTX # -DUSE_NOWAIT
LDFLAGS=-L$(ROCM_RUNTIME_PATH)/lib
LDLIBS=-lrocprofiler-sdk -lroctracer64 -lroctx64

PROFILE_CMD=rocprofv3 -o gpu_kernel_latency_trace --kernel-trace \
		--memory-copy-trace --memory-allocation-trace \
		--scratch-memory-trace --marker-trace --output-format pftrace --
endif

ifeq ($(COMP),nvidia)
FC=nvfortran
FCFLAGS=-g -O2 -mp=gpu -gpu=cc80 -fPIC -DUSE_NVTX # -DUSE_NOWAIT
LDFLAGS=-L$(CUDA_HOME)/targets/x86_64-linux/lib
LDLIBS=-lnvtx3interop

PROFILE_CMD=nsys profile -o gpu_kernel_latency_trace_results -t cuda,nvtx,openmp,openacc
endif

ifeq ($(COMP),gnu)
FC=gfortran
FCFLAGS=-g -O2 -fPIC 
LDFLAGS=
LDLIBS=
endif

EXE1=gpu_kernel_latency.x
all: $(EXE1)

OBJS1=gpu_kernel_latency.o gpu_kernel_latency_mod.o

gpu_kernel_latency.o: gpu_kernel_latency.F90 gpu_kernel_latency_mod.o
gpu_kernel_latency_mod.o: gpu_kernel_latency_mod.F90 

$(EXE1): $(OBJS1)
	$(FC) $(FCFLAGS) $^ $(LDFLAGS) $(LDLIBS) -o $@

%.o: %.F90
	$(FC) $(FCFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

run: $(EXE1)
	./$(EXE1)

prof: $(EXE1)
	$(PROFILE_CMD) ./$(EXE1)

.PHONY: clean

clean: 
	-/bin/rm -f $(EXE1) a.out *.o *~ *.mod *.so *.pftrace *.nsys-rep
