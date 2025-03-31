#-----------------------------------------------------------------------
# Makefile for OpenMP Driver Tests
#
# This Makefile demonstrates various compiler options for building
# OpenMP offloading applications targeting GPUs. It supports both
# AMDGCN (e.g., gfx targets) and NVPTX (e.g., sm_ targets) architectures.
#
# Usage:
# 1. Set the target GPU architecture using the LLVM_GPU_ARCH variable.
#    Example: export LLVM_GPU_ARCH=sm_30
# 2. Run `make run` to compile and execute the application.
#
# Targets:
# - `make`: Builds the application and generates intermediate files.
# - `make run`: Runs the compiled application.
# - `make clean`: Cleans up all generated files.
#
# Run `make help` to see additional options.
#-----------------------------------------------------------------------

# Define the test name and source file
TESTNAME = driver_tests
TESTSRC  = $(TESTNAME).c

# Determine the directory of the Makefile
mkfile_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ifneq ($(CURDIR)/,$(mkfile_dir))
  TESTSRC := $(mkfile_dir)$(TESTSRC)
endif

# Include helper script to find GPU and installation directory
include $(mkfile_dir)../../inc/find_gpu_and_install_dir.mk

# Compiler and flags
CC = $(LLVM_INSTALL_DIR)/bin/clang

# Base compiler flags for OpenMP
CFLAGS := -fopenmp

# Offload architecture flag
OAFLAG := --offload-arch=$(LLVM_GPU_ARCH)

# Enable verbose mode if VERBOSE=1
ifeq ($(VERBOSE),1)
  $(info    Compilation VERBOSE Mode ON)
  CFLAGS += -v
endif

# Allow additional flags to be appended
CFLAGS += $(EXTRA_CFLAGS)

#-----------------------------------------------------------------------
# Build target: Compile and link in one step, demonstrating various
# compiler options and generating intermediate files.
#-----------------------------------------------------------------------
$(TESTNAME): $(TESTSRC)
	@echo 
	@echo "------- DEMO VARIOUS COMPILER OPTIONS --------"
	@echo 
	@echo "1. Create HOST text LLVM IR:	$(TESTNAME)_host.ll"
	$(CC) $(CFLAGS) -emit-llvm -S $^ -o $(TESTNAME)_host.ll
	file $(TESTNAME)_host.ll 
	@echo 
	@echo "2. Create HOST LLVM bitcode:	$(TESTNAME)_host.bc"
	$(CC) $(CFLAGS) -emit-llvm -c $^ -o $(TESTNAME)_host.bc
	file $(TESTNAME)_host.bc
	@echo 
	@echo "3. Create HOST assembler:	$(TESTNAME)_host.s"
	$(CC) $(CFLAGS) -S $^ -o $(TESTNAME)_host.s
	file $(TESTNAME)_host.s
	@echo 
	@echo "4. Create DEVICE LLVM bitcode: $(TESTNAME)_device.bc"
	$(CC) $(CFLAGS) $(OAFLAG) --offload-device-only -emit-llvm -c $^ -o $(TESTNAME)_device.bc
	file $(TESTNAME)_device.bc
	@echo 
	@echo "5. Disassemble DEVICE LLVM bitcode to text: $(TESTNAME)_device.ll"
	$(LLVM_INSTALL_DIR)/bin/llvm-dis $(TESTNAME)_device.bc -o $(TESTNAME)_device.ll
	file $(TESTNAME)_device.ll
	@echo 
	@echo "6. Create application binary:	$(TESTNAME)"
	$(CC) $(CFLAGS) $(OAFLAG) $(LFLAGS) $^ -o $@
	file $(TESTNAME)
	@echo 

#-----------------------------------------------------------------------
# Run target: Execute the compiled application.
#-----------------------------------------------------------------------
run: $(TESTNAME)
	./$(TESTNAME)

#-----------------------------------------------------------------------
# Clean target: Remove all generated files.
#-----------------------------------------------------------------------
clean:
	rm -f $(TESTNAME) obin *.i *.ii *.bc *.lk a.out-* *.ll *.s *.o *.cubin
