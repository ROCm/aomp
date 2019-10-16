#-----------------------------------------------------------------------
#
#  Makefile: RAJA Makefile for amdgcn target.
#            export AOMP_GPU=gfx900
#            make run
#
#  Run "make help" to see other options for this Makefile

TESTNAME = raja_vecadd
TESTSRC  = raja_vecadd.cpp

# --- Standard Makefile check for AOMP installation ---
ifeq ("$(wildcard $(AOMP))","")
  ifneq ($(AOMP),)
    $(warning AOMP not found at $(AOMP))
  endif
  AOMP = $(HOME)/rocm/aomp
  ifeq ("$(wildcard $(AOMP))","")
    $(warning AOMP not found at $(AOMP))
    AOMP = /opt/rocm/aomp
    ifeq ("$(wildcard $(AOMP))","")
      $(warning AOMP not found at $(AOMP))
      $(error Please install AOMP or correctly set env-var AOMP)
    endif
  endif
endif
# --- End Standard Makefile check for AOMP installation ---
INSTALLED_GPU  = $(shell $(AOMP)/bin/mygpu -d gfx900)# Default AOMP_GPU is gfx900 which is vega
AOMP_GPU       ?= $(INSTALLED_GPU)
AOMP_CPUTARGET ?= x86_64-pc-linux-gnu
CC              = $(AOMP)/bin/clang++
AOMP_GPUTARGET = amdgcn-amd-amdhsa
export PATH := $(AOMP)/bin:$(PATH)

# Sorry, clang openmp requires these complex options
CPPFLAGS = -w -O3 -target $(AOMP_CPUTARGET) -fopenmp -fopenmp-targets=$(AOMP_GPUTARGET) -Xopenmp-target=$(AOMP_GPUTARGET) -march=$(AOMP_GPU) -I../../../raja/include -I./raja_build/include

ifeq ($(OFFLOAD_DEBUG),1)
  $(info    DEBUG Mode ON)
  CCENV  = env LIBRARY_PATH=$(AOMP)/lib-debug
  RUNENV = LIBOMPTARGET_DEBUG=1
endif

ifeq ($(VERBOSE),1)
  $(info    Compilation VERBOSE Mode ON)
  CPPFLAGS += -v
endif

ifeq ($(TEMPS),1)
  $(info    Compilation and linking save-temp Mode ON)
  CPPFLAGS += -save-temps -fopenmp-dump-offload-linker-script
endif

CPPFLAGS += $(EXTRA_CPPFLAGS)

# ----- Demo compile and link in one step, no object code saved

$(TESTNAME): raja_build $(TESTSRC)
	$(CCENV) $(CC) $(CPPFLAGS) $(LFLAGS) $(TESTSRC) -o $@

run: $(TESTNAME) 
	$(RUNENV) ./$(TESTNAME)

raja_build:
	patch -p1 -s -d ../../../raja < raja.patch; \
	patch -p1 -s -d ../../../raja/blt < blt.patch; \
	mkdir -p raja_build; cd raja_build; \
	cmake  -DOpenMP_C_FLAGS="-w;--target=x86_64-pc-linux-gnu;-fopenmp;-fopenmp-targets=amdgcn-amd-amdhsa;-Xopenmp-target=amdgcn-amd-amdhsa;-march=gfx906" \
	-DOpenMP_CXX_FLAGS="-w;--target=x86_64-pc-linux-gnu;-fopenmp;-fopenmp-targets=amdgcn-amd-amdhsa;-Xopenmp-target=amdgcn-amd-amdhsa;-march=gfx906" \
	-DENABLE_TARGET_OPENMP=On \
        -DENABLE_CUDA=Off \
        -DENABLE_CLANG_CUDA=Off \
        -DCMAKE_EXE_LINKER_FLAGS="" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_POSITION_INDEPENDENT_CODE=FALSE \
        -Wno-dev \
	-DRAJA_ENABLE_OPENMP=On \
        ../../../../raja; \
	make -j8; \
	cd ..

#  ----   Demo compile and link in two steps, object saved
$(TESTNAME).o: biuld_$(TESTSRC)
	$(CCENV) $(CC) -c $(CPPFLAGS) $^ -o $@

obin:	$(TESTNAME).o
	$(CCENV) $(CC) $(CPPFLAGS) $(LFLAGS) $^ -o $@

run_obin: obin
	$(RUNENV) ./obin

help:
	@echo
	@echo "Source[s]:		$(TESTSRC)"
	@echo "Application binary:    	$(TESTNAME)"
	@echo "Target GPU:		$(AOMP_GPU)"
	@echo "Target triple:		$(AOMP_GPUTARGET)"
	@echo "AOMP compiler: 		$(CC)"
	@echo "Compile flags:		$(CFLAGS)"
ifeq (sm_,$(findstring sm_,$(AOMP_GPU)))
	@echo "CUDA installation:	$(CUDA)"
endif
	@echo
	@echo "This Makefile supports these targets:"
	@echo
	@echo " make			// Builds $(TESTNAME) and RAJA "
	@echo " make run		// Executes $(TESTNAME) "
	@echo
	@echo " make clean              // Deletes test build and RAJA build "
	@echo " make help"
	@echo
	@echo "Environment variables used by this Makefile:"
	@echo "  AOMP_GPU=<GPU>       Target GPU, e.g sm_30, default=gfx900. To build for"
	@echo "                       Nvidia GPUs, set AOMP_GPU=sm_60 or appropriate sm_"
	@echo "  AOMP=<dir>           AOMP install dir, default=/opt/rocm/aomp"
	@echo "  EXTRA_CFLAGS=<args>  extra arguments for compiler"
	@echo "  OFFLOAD_DEBUG=n      if n=1, compile and run in Debug mode"
	@echo "  VERBOSE=n            if n=1, add verbose output"
	@echo "  TEMPS=1              do not delete intermediate files"
ifeq (sm_,$(findstring sm_,$(AOMP_GPU)))
	@echo "  CUDA=<dir>           CUDA install dir, default=/usr/local/cuda"
endif
	@echo

# Cleanup anything this makefile can create
clean:
	rm -rf $(TESTNAME) obin *.i *.ii *.bc *.lk a.out-* *.ll *.s *.o *.cubin raja_build
