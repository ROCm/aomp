#-----------------------------------------------------------------------
#
#  Makefile: RAJA Makefile for amdgcn target.
#            export LLVM_GPU_ARCH=gfx900
#            make run
#
#  Run "make help" to see other options for this Makefile

TESTNAME = vecadd
TESTSRC  = vecadd.cpp

mkfile_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ifneq ($(CURDIR)/,$(mkfile_dir))
  TESTSRC := $(mkfile_dir)$(TESTSRC)
endif
include $(mkfile_dir)../../Makefile.find_gpu_and_install_dir

CC = $(LLVM_INSTALL_DIR)/bin/clang++

export PATH := $(LLVM_INSTALL_DIR)/bin:$(PATH)
RAJA_BUILD_PREFIX ?= $(HOME)
RAJA_BUILD_OMP_DIR ?= $(RAJA_BUILD_PREFIX)/raja_build_omp.$(LLVM_GPU_ARCH)
RAJA_BUILD_HIP_DIR ?= $(RAJA_BUILD_PREFIX)/raja_build_hip.$(LLVM_GPU_ARCH)

ifeq ($(RAJA_USE_HIP),1)
  RAJA_BUILD_DIR = $(RAJA_BUILD_HIP_DIR)
else
# This is the default
  RAJA_BUILD_DIR = $(RAJA_BUILD_OMP_DIR)
endif

RAJA_SRC = $(HOME)/git/aomp/raja

# Sorry, clang openmp requires these complex options
CPPFLAGS = -w -O3 -fopenmp --offload-arch=$(LLVM_GPU_ARCH)$(AOMP_TARGET_FEATURES) -I$(RAJA_SRC)/include  -I$(RAJA_BUILD_DIR)/include -I$(RAJA_SRC)/tpl/camp/include

ifeq ($(OFFLOAD_DEBUG),1)
  $(info    DEBUG Mode ON)
  CCENV  = env LIBRARY_PATH=$(LLVM_INSTALL_DIR)/lib-debug
  RUNENV = LIBOMPTARGET_DEBUG=1
endif

ifeq ($(VERBOSE),1)
  $(info    Compilation VERBOSE Mode ON)
  CPPFLAGS += -v
endif

ifeq ($(TEMPS),1)
  $(info    Compilation and linking save-temp Mode ON)
  CPPFLAGS += -save-temps 
endif

CPPFLAGS += $(EXTRA_CPPFLAGS)
LIBS = $(RAJA_BUILD_DIR)/lib/libRAJA.a
LLD     = $(LLVM_INSTALL_DIR)/bin/clang++
LFLAGS          =

# ----- Demo compile and link in one step, no object code saved

$(TESTNAME): $(RAJA_BUILD_DIR) $(TESTSRC)
	$(CCENV) $(CC) $(CPPFLAGS) $(LFLAGS) $(TESTSRC) $(LIBS) -o $@

run: $(TESTNAME) 
	$(RUNENV) ./$(TESTNAME)

#  This is default build
$(RAJA_BUILD_OMP_DIR):
	$(LLVM_INSTALL_DIR)/bin/raja_build.sh

$(RAJA_BUILD_HIP_DIR):
	$(LLVM_INSTALL_DIR)/bin/raja_build.sh hip $(RAJA_BUILD_HIP_DIR)

help:
	@echo
	@echo "Source[s]:		$(TESTSRC)"
	@echo "Application binary:    	$(TESTNAME)"
	@echo "Target GPU:		$(LLVM_GPU_ARCH)"
	@echo "Compiler: 		$(CC)"
	@echo "Compile flags:		$(CFLAGS)"
	@echo "RAJA build prefix:	$(RAJA_BUILD_PREFIX)"
ifeq (sm_,$(findstring sm_,$(LLVM_GPU_ARCH)))
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
	@echo "  LLVM_GPU_ARCH=<GPU>       Target GPU, e.g sm_30, default=gfx900. To build for"
	@echo "                       Nvidia GPUs, set LLVM_GPU_ARCH=sm_60 or appropriate sm_"
	@echo "  LLVM_INSTALL_DIR=<dir>"
	@echo "  EXTRA_CFLAGS=<args>  extra arguments for compiler"
	@echo "  OFFLOAD_DEBUG=n      if n=1, compile and run in Debug mode"
	@echo "  VERBOSE=n            if n=1, add verbose output"
	@echo "  TEMPS=1              do not delete intermediate files"
ifeq (sm_,$(findstring sm_,$(LLVM_GPU_ARCH)))
	@echo "  CUDA=<dir>           CUDA install dir, default=/usr/local/cuda"
endif
	@echo

# Cleanup anything this makefile can create
clean:
	rm -rf $(TESTNAME) obin *.i *.ii *.bc *.lk a.out-* *.ll *.s *.o *.cubin

clean_raja:
	rm -rf $(RAJA_BUILD_DIR)
