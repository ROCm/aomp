#-----------------------------------------------------------------------
# Makefile for OpenMP offloading example using declare_variant_if.
# This Makefile demonstrates building and running OpenMP programs
# targeting both AMD GPUs (amdgcn) and NVIDIA GPUs (nvptx).
#
# Usage:
#   Set the target GPU architecture using LLVM_GPU_ARCH.
#   Example: export LLVM_GPU_ARCH=sm_30
#   Run `make run` to build and execute the program.
#
# Targets:
#   - make: Builds the binary.
#   - make run: Runs the binary.
#   - make clean: Cleans up generated files.
#   - make help: Displays help information.
#-----------------------------------------------------------------------

TESTNAME = declare_variant_if
TESTSRC  = declare_variant_if.c

# Determine the directory of the Makefile
mkfile_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ifneq ($(CURDIR)/,$(mkfile_dir))
  TESTSRC := $(mkfile_dir)$(TESTSRC)
endif

# Include common scripts for GPU detection and installation paths
include $(mkfile_dir)../../inc/find_gpu_and_install_dir.mk

# Compiler and flags setup for OpenMP offloading
CC = $(LLVM_INSTALL_DIR)/bin/clang
# OpenMP 5.0 requires -fopenmp-version=50 and --offload-arch for GPU targets
CFLAGS = -O3 -fopenmp -fopenmp-version=50 --offload-arch=$(LLVM_GPU_ARCH)$(AOMP_TARGET_FEATURES)

# Debug mode setup
ifeq ($(OFFLOAD_DEBUG),1)
  $(info    DEBUG Mode ON)
  CCENV  = env LIBRARY_PATH=$(LLVM_INSTALL_DIR)/lib-debug
  RUNENV = LIBOMPTARGET_DEBUG=1
endif

# Verbose mode setup
ifeq ($(VERBOSE),1)
  $(info    Compilation VERBOSE Mode ON)
  CFLAGS += -v
endif

# Save intermediate files for debugging
ifeq ($(TEMPS),1)
  $(info    Compilation and linking save-temp Mode ON)
  CFLAGS += -save-temps 
endif

# Add any extra flags passed via EXTRA_CFLAGS
CFLAGS += $(EXTRA_CFLAGS)

# Build target: Compile and link in one step
$(TESTNAME): $(TESTSRC)
	$(CCENV) $(CC) $(CFLAGS) $(LFLAGS) $^ -o $@

# Run target: Execute the compiled binary
run: $(TESTNAME)
	$(RUNENV) ./$(TESTNAME)

# Include additional help and utility scripts
include $(mkfile_dir)../../inc/help.mk
include $(mkfile_dir)../../inc/obin.mk

# Cleanup target: Remove generated files
clean:
	rm -f $(TESTNAME) obin *.i *.ii *.bc *.lk a.out-* *.ll *.s *.o *.cubin
