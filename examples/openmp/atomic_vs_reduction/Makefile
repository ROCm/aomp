#-----------------------------------------------------------------------
#
# This Makefile is designed to build and run OpenMP offloading examples
# for both AMD (amdgcn) and NVIDIA (nvptx) GPU targets. 
#
# Usage:
# 1. Set the target GPU architecture using the LLVM_GPU_ARCH variable.
#    Example: export LLVM_GPU_ARCH=sm_30
# 2. Run `make` to build the example.
# 3. Run `make run` to execute the example.
#
# Options:
# - OFFLOAD_DEBUG=1: Enable debug mode for offloading.
# - VERBOSE=1: Enable verbose compilation output.
# - TEMPS=1: Save intermediate compilation and linking files.
#
# Cleanup:
# Run `make clean` to remove all generated files.
#
# Additional Help:
# Run `make help` to see other options for this Makefile.
#-----------------------------------------------------------------------

# Get the directory of the current Makefile
mkfile_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Include helper scripts for GPU detection and installation directory
include $(mkfile_dir)../../inc/find_gpu_and_install_dir.mk

# Test name and source file
TESTNAME = atomic_vs_reduction
TESTSRC = atomic_vs_reduction.c

# Adjust the source file path if the Makefile is invoked from a different directory
ifneq ($(CURDIR)/,$(mkfile_dir))
  TESTSRC := $(mkfile_dir)$(TESTSRC)
endif

# Compiler and flags
CC = $(LLVM_INSTALL_DIR)/bin/clang
CFLAGS = -O3 -fopenmp --offload-arch=$(LLVM_GPU_ARCH)$(AOMP_TARGET_FEATURES)

# Enable debug mode if OFFLOAD_DEBUG is set
ifeq ($(OFFLOAD_DEBUG),1)
  $(info    DEBUG Mode ON)
  CCENV  = env LIBRARY_PATH=$(LLVM_INSTALL_DIR)/lib-debug
  RUNENV = LIBOMPTARGET_DEBUG=1
endif

# Enable verbose mode if VERBOSE is set
ifeq ($(VERBOSE),1)
  $(info    Compilation VERBOSE Mode ON)
  CFLAGS += -v
endif

# Add a macro definition if the compiler is clang
ifeq ($(LLVM_COMPILER_NAME),clang)
  CFLAGS += -D__LLVM_COMPILER_NAME_IS_CLANG__
endif

# Save intermediate files if TEMPS is set
ifeq ($(TEMPS),1)
  $(info    Compilation and linking save-temp Mode ON)
  CFLAGS += -save-temps 
endif

# Append any extra flags passed via EXTRA_CFLAGS
CFLAGS += $(EXTRA_CFLAGS)

# Build target
$(TESTNAME): $(TESTSRC)
	$(CCENV) $(CC) $(CFLAGS) $(LFLAGS) $^ -o $@

# Run target
run: $(TESTNAME)
	$(RUNENV) ./$(TESTNAME)

# Include additional helper scripts
include $(mkfile_dir)../../inc/obin.mk
include $(mkfile_dir)../../inc/help.mk

# Cleanup target
clean:
	rm -f $(TESTNAME) obin *.i *.ii *.bc *.lk a.out-* *.ll *.s *.o *.cubin
