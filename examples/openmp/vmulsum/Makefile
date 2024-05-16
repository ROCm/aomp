#-----------------------------------------------------------------------
#  ROCM_LLVM:       Defaults to /opt/rocm/llvm
#  ROCM_GPU:        Auto detected via rocm_agent_enumerator
#  ROCM_GPUTARGET:  Defaults to amdgcn-amd-amdhsa
#
#  Makefile: Cuda clang demo Makefile for both amdgcn and nvptx targets.
#            amdgcn GPU targets begin with "gfx". nvptx targets begin
#            with sm_.  Example: To build and run on k4000 do this:
#
#            export LLVM_GPU_ARCH=sm_30
#            make run
#
#  Run "make help" to see other options for this Makefile

TESTNAME = vmulsum
S1 = main.c
S2 = vsum.c 
S3 = vmul.c

mkfile_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ifneq ($(CURDIR)/,$(mkfile_dir))
  TESTSRC := $(mkfile_dir)$(S1) $(mkfile_dir)$(S2) $(mkfile_dir)$(S3) 
else
  TESTSRC := $(S1) $(S2) $(S3) 
endif
include $(mkfile_dir)../../Makefile.find_gpu_and_install_dir

CC = $(LLVM_INSTALL_DIR)/bin/clang

# Sorry, clang openmp requires these complex options
CFLAGS = -O3 -fopenmp --offload-arch=$(LLVM_GPU_ARCH)$(AOMP_TARGET_FEATURES)

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

#  ----   Demo compile and link in two steps, object saved
main.o: main.c
	$(CCENV) $(CC) -c $(CFLAGS) $^ -o $@
vsum.o: vsum.c
	$(CCENV) $(CC) -c $(CFLAGS) $^ -o $@
vmul.o: vmul.c
	$(CCENV) $(CC) -c $(CFLAGS) $^ -o $@
obin:	main.o vsum.o vmul.o
	$(CCENV) $(CC) $(CFLAGS) $(LFLAGS) $^ -o $@

run_obin: obin
	$(RUNENV) ./obin

help:
	@echo
	@echo "Source[s]:		$(TESTSRC)"
	@echo "Application binary:    	$(TESTNAME)"
	@echo "Target GPU:		$(LLVM_GPU_ARCH)"
	@echo "Compiler: 		$(CC)"
	@echo "Compile flags:		$(CFLAGS)"
	@echo
	@echo "This Makefile supports these targets:"
	@echo
	@echo " make			// Builds $(TESTNAME) "
	@echo " make run		// Executes $(TESTNAME) "
	@echo
	@echo " make main.o"
	@echo " make vsum.o"
	@echo " make vmul.o"
	@echo " make obin		// Link object files to build binary "
	@echo " make run_obin		// Execute obin "
	@echo
	@echo " make clean"
	@echo " make help"
	@echo
	@echo "Environment variables used by this Makefile:"
	@echo "  LLVM_GPU_ARCH=<GPU>       Target GPU, e.g sm_30, default=gfx900. To build for"
	@echo "                       Nvidia GPUs, set LLVM_GPU_ARCH=sm_60 or appropriate sm_"
	@echo "  LLVM_INSTALL_DIR   "
	@echo "  EXTRA_CFLAGS=<args>  extra arguments for compiler"
	@echo "  OFFLOAD_DEBUG=n      if n=1, compile and run in Debug mode"
	@echo "  VERBOSE=n            if n=1, add verbose output"
	@echo "  TEMPS=1              do not delete intermediate files"
	@echo

# Cleanup anything this makefile can create
clean:
	rm -f $(TESTNAME) obin *.i *.ii *.bc *.lk a.out-* *.ll *.s *.o *.cubin
