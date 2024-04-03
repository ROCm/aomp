#-----------------------------------------------------------------------
#
#  Makefile: Cuda clang demo Makefile for both amdgcn and nvptx targets.
#            amdgcn GPU targets begin with "gfx". nvptx targets begin
#            with sm_.  Example: To build and run on k4000 do this:
#
#            export LLVM_GPU_ARCH=sm_30
#            make run
#
#  Run "make help" to see other options for this Makefile

TESTNAME = declare_variant_if 
TESTSRC  = declare_variant_if.c

mkfile_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ifneq ($(CURDIR)/,$(mkfile_dir))
  TESTSRC := $(mkfile_dir)$(TESTSRC)
endif
include $(mkfile_dir)../../Makefile.find_gpu_and_install_dir

CC = $(LLVM_INSTALL_DIR)/bin/clang
# Sorry, clang openmp requires these complex options
CFLAGS = -O3 -fopenmp -fopenmp-version=50 --offload-arch=$(LLVM_GPU_ARCH)$(AOMP_TARGET_FEATURES)

ifeq ($(OFFLOAD_DEBUG),1)
  $(info    DEBUG Mode ON)
  CCENV  = env LIBRARY_PATH=$(LLVM_INSTALL_DIR)/lib-debug
  RUNENV = LIBOMPTARGET_DEBUG=1
endif

ifeq ($(VERBOSE),1)
  $(info    Compilation VERBOSE Mode ON)
  CFLAGS += -v
endif

ifeq ($(TEMPS),1)
  $(info    Compilation and linking save-temp Mode ON)
  CFLAGS += -save-temps 
endif

CFLAGS += $(EXTRA_CFLAGS)

# ----- Demo compile and link in one step, no object code saved
$(TESTNAME): $(TESTSRC)
	$(CCENV) $(CC) $(CFLAGS) $(LFLAGS) $^ -o $@

run: $(TESTNAME)
	$(RUNENV) ./$(TESTNAME)

help:
	@echo
	@echo "Source[s]:		$(TESTSRC)"
	@echo "Application binary:    	$(TESTNAME)"
	@echo "Target GPU:		$(LLVM_GPU_ARCH)"
	@echo "Target triple:		$(LLVM_GPU_TRIPLE)"
	@echo "Compiler: 		$(CC)"
	@echo "Compile flags:		$(CFLAGS)"
ifeq (sm_,$(findstring sm_,$(LLVM_GPU_ARCH)))
	@echo "CUDA installation:	$(CUDA)"
endif
	@echo
	@echo "This Makefile supports these targets:"
	@echo
	@echo " make			// Builds $(TESTNAME) "
	@echo " make run		// Executes $(TESTNAME) "
	@echo
	@echo " make clean"
	@echo " make help"
	@echo
	@echo "Environment variables used by this Makefile:"
	@echo "  LLVM_GPU_ARCH=<GPU>       Target GPU, e.g sm_30, default=gfx900. To build for"
	@echo "                       Nvidia GPUs, set LLVM_GPU_ARCH=sm_60 or appropriate sm_"
	@echo "  LLVM_INSTALL_DIR     Compiler install dir"
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
	rm -f $(TESTNAME) obin *.i *.ii *.bc *.lk a.out-* *.ll *.s *.o *.cubin
