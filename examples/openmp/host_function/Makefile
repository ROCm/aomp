#-----------------------------------------------------------------------
#  ROCM_LLVM:       Defaults to /opt/rocm/llvm
#  ROCM_GPU:        Auto detected via rocm_agent_enumerator
#  ROCM_GPUTARGET:  Defaults to amdgcn-amd-amdhsa
#
#  make run
#
#  Run "make help" to see other options for this Makefile

TESTNAME = host_function
TESTSRC  = host_function.c

UNAMEP = $(shell uname -p)
ROCM_CPUTARGET = $(UNAMEP)-pc-linux-gnu
ifeq ($(UNAMEP),ppc64le)
  ROCM_CPUTARGET = ppc64le-linux-gnu
endif

ROCM_LLVM ?= /opt/rocm/llvm
ROCM_GPUTARGET ?= amdgcn-amd-amdhsa

INSTALLED_GPU  = $(shell $(ROCM_LLVM)/../bin/rocm_agent_enumerator | grep -m 1 -E gfx[^0]{1})
ROCM_GPU       ?= $(INSTALLED_GPU)
CC              = $(ROCM_LLVM)/bin/clang

ROCM_GPUTARGET = amdgcn-amd-amdhsa

# Sorry, clang openmp requires these complex options
CFLAGS = -O3 -target $(ROCM_CPUTARGET) -fopenmp -fopenmp-targets=$(ROCM_GPUTARGET) -Xopenmp-target=$(ROCM_GPUTARGET) -march=$(ROCM_GPU)

ifeq ($(OFFLOAD_DEBUG),1)
  $(info    DEBUG Mode ON)
  CCENV  = env LIBRARY_PATH=$(ROCM_LLVM)/lib-debug
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
$(TESTNAME).o: $(TESTSRC)
	$(CCENV) $(CC) -c $(CFLAGS) $^ -o $@

obin:	$(TESTNAME).o
	$(CCENV) $(CC) $(CFLAGS) $(LFLAGS) $^ -o $@

run_obin: obin
	$(RUNENV) ./obin

help:
	@echo
	@echo "Source[s]:		$(TESTSRC)"
	@echo "Application binary:    	$(TESTNAME)"
	@echo "Target GPU:		$(ROCM_GPU)"
	@echo "Target triple:		$(ROCM_GPUTARGET)"
	@echo "ROCM compiler: 		$(CC)"
	@echo "Compile flags:		$(CFLAGS)"
	@echo
	@echo "This Makefile supports these targets:"
	@echo
	@echo " make			// Builds $(TESTNAME) "
	@echo " make run		// Executes $(TESTNAME) "
	@echo
	@echo " make $(TESTNAME).o		// build object file "
	@echo " make obin		// Link object file to build binary "
	@echo " make run_obin		// Execute obin "
	@echo
	@echo " make clean"
	@echo " make help"
	@echo
	@echo "Environment variables used by this Makefile:"
	@echo "  ROCM_GPU=<GPU>       Target GPU, e.g gfx906"
	@echo "  ROCM_LLVM=<dir>      Where llvm is located"
	@echo "  EXTRA_CFLAGS=<args>  extra arguments for compiler"
	@echo "  OFFLOAD_DEBUG=n      if n=1, compile and run in Debug mode"
	@echo "  VERBOSE=n            if n=1, add verbose output"
	@echo "  TEMPS=1              do not delete intermediate files"
	@echo

# Cleanup anything this makefile can create
clean:
	rm -f $(TESTNAME) obin *.i *.ii *.bc *.lk a.out-* *.ll *.s *.o *.cubin
