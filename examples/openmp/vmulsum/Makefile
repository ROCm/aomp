#-----------------------------------------------------------------------
#
#  Makefile: demo Makefile for compiler example vmulsum
#
#-----------------------------------------------------------------------

TESTNAME = vmulsum
S1 = main.c
S2 = vsum.c 
S3 = vmul.c

CFLAGS ?=
mkfile_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
include $(mkfile_dir)../../inc/find_gpu_and_install_dir.mk
ifneq ($(CURDIR)/,$(mkfile_dir))
  _in_dir := $(mkfile_dir)
else
  _in_dir :=
endif
# All inputs from example dir should be prefixed with $_in_dir
GPUFLAGS ?= --offload-arch=$(LLVM_GPU_ARCH)$(AOMP_TARGET_FEATURES)

# Collect all the source files
TESTSRCS := $(addprefix $(_in_dir), $(S1) $(S2) $(S3))

CC = $(LLVM_INSTALL_DIR)/bin/clang
# Set clang openmp options
CFLAGS = -O3 -fopenmp

# If we use trunk, there might not be any amd device libs and we don't need
# them for this example since it's not using any math libs.
ifeq ($(LLVM_COMPILER_NAME),clang)
  CFLAGS += -lomptarget.devicertl -nogpulib
endif

ifeq ($(OFFLOAD_DEBUG),1)
  $(info    # DEBUG Mode ON)
  CCENV  = env LIBRARY_PATH=$(LLVM_INSTALL_DIR)/lib-debug
  RUNENV += LIBOMPTARGET_DEBUG=1
endif

ifeq ($(VERBOSE),1)
  $(info    # Compilation VERBOSE Mode ON)
  CFLAGS += -v
endif

ifeq ($(TEMPS),1)
  $(info    # Compilation and linking save-temp Mode ON)
  CFLAGS += -save-temps
endif

CFLAGS += $(EXTRA_CFLAGS) $(GPUFLAGS)

# ----- Demo compile and link in one step, no object code saved
$(TESTNAME): $(TESTSRCS)
	$(CCENV) $(CC) $(CFLAGS) $(LFLAGS) $^ -o $@

run: $(TESTNAME)
	$(RUNENV) ./$(TESTNAME)

#  ----   Demo compile and link in two steps, object saved
main.o: $(_in_dir)main.c
	$(CCENV) $(CC) -c $(CFLAGS) $^ -o $@
vsum.o: $(_in_dir)vsum.c
	$(CCENV) $(CC) -c $(CFLAGS) $^ -o $@
vmul.o: $(_in_dir)vmul.c
	$(CCENV) $(CC) -c $(CFLAGS) $^ -o $@
obin:	main.o vsum.o vmul.o
	$(CCENV) $(CC) $(CFLAGS) $(LFLAGS) $^ -o $@

run_obin: obin
	$(RUNENV) ./obin

help:
	@echo
	@echo "Source[s]:               $(TESTSRCS)"
	@echo "Application binary:      $(TESTNAME)"
	@echo "Target GPU:              $(LLVM_GPU_ARCH)"
	@echo "Compiler:                $(CC) ($(LLVM_COMPILER_NAME))"
	@echo "Compile flags:           $(CFLAGS)"
	@echo
	@echo "This Makefile supports these targets:"
	@echo
	@echo " make                    // Build $(TESTNAME) "
	@echo " make run                // Execute $(TESTNAME) "
	@echo
	@echo " make main.o"
	@echo " make vsum.o"
	@echo " make vmul.o"
	@echo " make obin               // Link object files to build binary "
	@echo " make run_obin           // Execute obin "
	@echo
	@echo " make clean"
	@echo " make help"
	@echo
	@echo "Environment variables used by this Makefile:"
	@echo "  LLVM_GPU_ARCH=<GPU>    Target GPU, e.g gfx90a (default: autodetect). To build for"
	@echo "                         Nvidia GPUs, set LLVM_GPU_ARCH=sm_60 or appropriate sm_"
	@echo "  LLVM_INSTALL_DIR=<directory>    LLVM installation directory"
	@echo "  EXTRA_CFLAGS=<args>    extra arguments for compiler"
	@echo "  OFFLOAD_DEBUG=n        if n=1, compile and run in Debug mode"
	@echo "  VERBOSE=n              if n=1, add verbose output"
	@echo "  TEMPS=n                if n=1, do not delete intermediate files"
	@echo

# Cleanup anything this makefile can create
clean:
	rm -f $(TESTNAME) obin *.i *.ii *.bc *.lk a.out-* *.ll *.s *.o *.cubin
