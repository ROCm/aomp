#-----------------------------------------------------------------------
#
#  Makefile: RAJA Makefile for amdgcn target.
#            export AOMP_GPU=gfx900
#            make run
#
#  Run "make help" to see other options for this Makefile

TESTNAME = vecadd
TESTSRC  = vecadd.cpp

UNAMEP = $(shell uname -p)
AOMP_CPUTARGET = $(UNAMEP)-pc-linux-gnu
ifeq ($(UNAMEP),ppc64le)
  AOMP_CPUTARGET = ppc64le-linux-gnu
endif
# --- Standard Makefile check for AOMP installation ---
ifeq ("$(wildcard $(AOMP))","")
  ifneq ($(AOMP),)
    $(warning AOMP not found at $(AOMP))
  endif
  AOMP = $(HOME)/rocm/aomp
  ifeq ("$(wildcard $(AOMP))","")
    $(warning AOMP not found at $(AOMP))
    AOMP = /usr/lib/aomp
    ifeq ("$(wildcard $(AOMP))","")
      $(warning AOMP not found at $(AOMP))
      $(error Please install AOMP or correctly set env-var AOMP)
    endif
  endif
endif
# --- End Standard Makefile check for AOMP installation ---
INSTALLED_GPU  = $(shell $(AOMP)/bin/mygpu -d gfx900)# Default AOMP_GPU is gfx900 which is vega
AOMP_GPU       ?= $(INSTALLED_GPU)
ifeq ($(RAJA_USE_HIP),1)
CC 		= $(AOMP)/bin/hipcc
else
CC              = $(AOMP)/bin/clang++
endif
AOMP_GPUTARGET = amdgcn-amd-amdhsa
export PATH := $(AOMP)/bin:$(PATH)
RAJA_BUILD_PREFIX ?= $(HOME)
RAJA_BUILD_OMP_DIR ?= $(RAJA_BUILD_PREFIX)/raja_build_omp.$(AOMP_GPU)
RAJA_BUILD_HIP_DIR ?= $(RAJA_BUILD_PREFIX)/raja_build_hip.$(AOMP_GPU)

ROCPRIM_BUILD_PREFIX ?= $(HOME)
ROCPRIM_BUILD_DIR=${ROCPRIM_BUILD_DIR:-$ROCPRIM_BUILD_PREFIX/rocprim_build}

ifeq ($(RAJA_USE_HIP),1)
  RAJA_BUILD_DIR = $(RAJA_BUILD_HIP_DIR)
else
# This is the default
  RAJA_BUILD_DIR = $(RAJA_BUILD_OMP_DIR)
endif

RAJA_SRC = $(HOME)/git/aomp/raja

# Sorry, clang openmp requires these complex options
ifeq ($(RAJA_USE_HIP),1)
CPPFLAGS = -DRAJA_USE_HIP -c -O2 -I$(RAJA_SRC)/include  -I$(RAJA_BUILD_DIR)/include -I$(RAJA_SRC)/tpl/camp/include
else
CPPFLAGS = -w -O3 -target $(AOMP_CPUTARGET) -fopenmp -fopenmp-targets=$(AOMP_GPUTARGET) -Xopenmp-target=$(AOMP_GPUTARGET) -march=$(AOMP_GPU) -I$(RAJA_SRC)/include  -I$(RAJA_BUILD_DIR)/include -I$(RAJA_SRC)/tpl/camp/include
endif

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
LDHELPER	= $(AOMP)/bin/hipcc_cmake_linker_helper
LIBS = $(RAJA_BUILD_DIR)/lib/libRAJA.a
LLD 		= $(AOMP)/bin/clang++
LFLAGS          =

# ----- Demo compile and link in one step, no object code saved

ifeq ($(RAJA_USE_HIP),1)
$(TESTNAME): $(TESTNAME).o
	$(LDHELPER) $(LLD) $(TESTNAME).o $(LIBS) -o $@
else
$(TESTNAME): $(RAJA_BUILD_DIR) $(TESTSRC)
	$(CCENV) $(CC) $(CPPFLAGS) $(LFLAGS) $(TESTSRC) $(LIBS) -o $@
endif

$(TESTNAME).o: $(RAJA_BUILD_DIR) $(AOMP)/rocprim $(TESTSRC)
	$(CCENV) $(CC) $(CPPFLAGS)  $(TESTSRC) -o $@

run: $(TESTNAME) 
	$(RUNENV) ./$(TESTNAME)

#  This is default build
$(RAJA_BUILD_OMP_DIR):
	$(AOMP)/bin/raja_build.sh

$(RAJA_BUILD_HIP_DIR):
	$(AOMP)/bin/raja_build.sh hip $(RAJA_BUILD_HIP_DIR)

$(AOMP)/rocprim:
	$(AOMP)/bin/build_rocprim.sh

help:
	@echo
	@echo "Source[s]:		$(TESTSRC)"
	@echo "Application binary:    	$(TESTNAME)"
	@echo "Target GPU:		$(AOMP_GPU)"
	@echo "Target triple:		$(AOMP_GPUTARGET)"
	@echo "AOMP compiler: 		$(CC)"
	@echo "Compile flags:		$(CFLAGS)"
	@echo "RAJA build prefix:	$(RAJA_BUILD_PREFIX)"
ifeq (sm_,$(findstring sm_,$(AOMP_GPU)))
	@echo "CUDA installation:	$(CUDA)"
endif
	@echo
	@echo "This Makefile supports these targets:"
	@echo
	@echo " make			// Builds $(TESTNAME) and RAJA "
	@echo " make run		// Executes $(TESTNAME) "
	@echo
	@echo " make clean              // Deletes test build  "
	@echo " make clean_raja         // Deletes RAJA build "
	@echo " make help"
	@echo
	@echo "Environment variables used by this Makefile:"
	@echo "  AOMP_GPU=<GPU>       Target GPU, e.g sm_30, default=gfx900. To build for"
	@echo "                       Nvidia GPUs, set AOMP_GPU=sm_60 or appropriate sm_"
	@echo "  AOMP=<dir>           AOMP install dir, default=/usr/lib/aomp"
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
	rm -rf $(TESTNAME) obin *.i *.ii *.bc *.lk a.out-* *.ll *.s *.o *.cubin

clean_raja:
	rm -rf $(RAJA_BUILD_DIR)
