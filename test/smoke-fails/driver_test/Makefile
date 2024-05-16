include ../../Makefile.defs

TESTNAME	= driver_test

AOMP_PATH 	?= $(HOME)/rocm/aomp
ROCM_PATH 	?= /opt/rocm
HSALIBDIR 	?= $(ROCM_PATH)/hsa/lib
OLARCH    	?= $(ROCM_PATH)/llvm/bin/offload-arch

FLANG       = $(AOMP)/bin/flang
CLANG		= $(AOMP)/bin/clang

TARGET     	= $(shell $(OLARCH))

FFLAGS     	= -c -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$(TARGET) -ffast-math -O3 -r8 -i8 -cpp -DPTRSIZE8 -DLONGINT8 -DUNDERSCORE -DGLOBAL_LONG_LONG
CCFLAGS		= -c -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$(TARGET) -ffast-math -O3 -DPTRSIZE8 -DLONGINT8 -DUNDERSCORE -DGLOBAL_LONG_LONG -g -DPREFIX=jl_
LD_FLAGS	= -O3 -fdisable-host-devmem -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$(TARGET) -ffast-math -lm

all: inc_arrayval.c.o main.c.o dec_arrayval.f95.o
	$(FLANG) $(LD_FLAGS) inc_arrayval.c.o dec_arrayval.f95.o main.c.o -o driver_test

inc_arrayval.c.o:
	$(CLANG) $(CCFLAGS) inc_arrayval.c -o $@

dec_arrayval.f95.o:
	$(FLANG) $(FFLAGS) dec_arrayval.f95 -o $@

main.c.o:
	$(CLANG) $(CCFLAGS) main.c -o $@

run: all
	./driver_test

.PHONY: clean
clean:
	rm *.o driver_test 2>/dev/null
