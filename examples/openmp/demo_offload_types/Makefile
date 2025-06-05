# --- Makefile for OpenMP Offload Types Demo ---
# This Makefile demonstrates the compilation and execution of binaries
# with different OpenMP offloading configurations, including no offload,
# host offload, and GPU offload.

TESTNAME = demo_offload_types
TESTSRC  = $(TESTNAME).c

# Determine the directory of the Makefile
mkfile_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ifneq ($(CURDIR)/,$(mkfile_dir))
  TESTSRC := $(mkfile_dir)$(TESTSRC)
endif
include $(mkfile_dir)../../inc/find_gpu_and_install_dir.mk

# Compiler and flags
CC = $(LLVM_INSTALL_DIR)/bin/clang
CFLAGS := -fopenmp
OAFLAG := --offload-arch=$(LLVM_GPU_ARCH)
OAQFLAG := --offload-arch=$(LLVM_GPU_ARCH):xnack+

# Enable verbose mode if requested
ifeq ($(VERBOSE),1)
  $(info    Compilation VERBOSE Mode ON)
  CFLAGS += -v
endif

CFLAGS += $(EXTRA_CFLAGS)

# --- Compilation and linking ---
# Compile and link binaries for different offload configurations
$(TESTNAME): $(TESTSRC)
	@echo "------- DEMO BUILDS OF VARIOUS OFFLOAD TYPES --------"
	@echo "1. Create binary that does not offload:	no-offload"
	$(CC) $(CFLAGS) $^ -o no-offload
ifeq ($(LLVM_COMPILER_NAME),clang)
	@echo "2 Skipping host-offload because this is broken in trunk"
else
	@echo "2. Create binary for host offloading:	host-offload"
	$(CC) $(CFLAGS) -fopenmp-targets=$(CLANG_HOST_TARGET) -Xopenmp-target=$(CLANG_HOST_TARGET) --march=znver1 $^ -o host-offload
endif
	@echo "3. Create binary that does GPU offload:	gpu-offload"
	$(CC) $(CFLAGS) $(OAFLAG) $^ -o gpu-offload 
ifneq (sm_,$(findstring sm_,$(LLVM_GPU_ARCH)))
	@echo "4. Create binary for GPU offload compiled with :xnack+"
	@echo "   The image will require HSA_XNACK=1 at runtime."
	$(CC) $(CFLAGS) $(OAQFLAG) $^ -o $(TESTNAME)
endif

# --- Execution ---
# Run the compiled binaries with different configurations
run: $(TESTNAME)
	@echo "------- DEMO DEFAULT EXECUTION -------"
	@echo "1. no-offload"
	./no-offload
ifeq ($(LLVM_COMPILER_NAME),clang)
	@echo "2 Skipping host-offload because this is broken in trunk"
else
	@echo "2. host-offload"
	./host-offload
endif
	@echo "3. gpu-offload"
	./gpu-offload
ifneq (sm_,$(findstring sm_,$(LLVM_GPU_ARCH)))
	@echo "4. GPU offload compiled with :xnack+,  Runtime fail will disable offloading."
ifeq ($(LLVM_COMPILER_NAME),AMD)
	@echo "   SKIPPING SINCE ROCm 6.1 has runtime fail here:  env HSA_XNACK=0 ./$(TESTNAME) "
else
	env HSA_XNACK=0 ./$(TESTNAME)
endif
	@echo "5. GPU offload compiled with xnack+.  This works on machines that support dynamic XNACK"
ifeq ($(LLVM_COMPILER_NAME),AMD)
	@echo "   SKIPPING SINCE ROCm 6.1 has runtime fail here:  env HSA_XNACK=1 ./$(TESTNAME) "
else
	env HSA_XNACK=1 ./$(TESTNAME)
endif
endif
	@echo "------- DEMO EXECUTION WITH OMP_TARGET_OFFLOAD=DISBLED -------"
	@echo "1. no-offload"
	env OMP_TARGET_OFFLOAD=DISABLED ./no-offload
ifneq ($(LLVM_COMPILER_NAME),clang)
	@echo "2. host-offload"
	env OMP_TARGET_OFFLOAD=DISABLED ./host-offload
endif
	@echo "3. gpu-offload"
	env OMP_TARGET_OFFLOAD=DISABLED ./gpu-offload
ifneq (sm_,$(findstring sm_,$(LLVM_GPU_ARCH)))
	@echo "4. GPU offload compiled with :xnack+ but run with HSA_XNACK=0"
	env OMP_TARGET_OFFLOAD=DISABLED HSA_XNACK=0 ./$(TESTNAME) 
	@echo "5. GPU offload compiled with :xnack+"
	env OMP_TARGET_OFFLOAD=DISABLED HSA_XNACK=1 ./$(TESTNAME) 
endif
	@echo "------- DEMO EXECUTION WITH OMP_TARGET_OFFLOAD=MANDATORY-------"
	@echo "1. no-offload"
	env OMP_TARGET_OFFLOAD=MANDATORY ./no-offload
ifneq ($(LLVM_COMPILER_NAME),clang)
	@echo "2. host-offload"
	env OMP_TARGET_OFFLOAD=MANDATORY ./host-offload
endif
	@echo "3. GPU offload"
	env OMP_TARGET_OFFLOAD=MANDATORY ./gpu-offload
ifneq (sm_,$(findstring sm_,$(LLVM_GPU_ARCH)))
	@echo "4. SKIPPING execution of binary compiled with xnack+ with OMP_TARGET_OFFLOAD=MANDATORY"
	@echo "   and run with HSA_XNACK=0 because this always creates expected fatal runtime error."
	@echo "5. GPU offload compiled with :xnack+ and run with HSA_XNACK=1"
ifeq ($(LLVM_COMPILER_NAME),AMD)
	@echo "   SKIPPING SINCE ROCm 6.1 has runtime fail here: env OMP_TARGET_OFFLOAD=MANDATORY HSA_XNACK=1 ./$(TESTNAME)"
else
	env OMP_TARGET_OFFLOAD=MANDATORY HSA_XNACK=1 ./$(TESTNAME)
endif
endif

# --- Cleanup ---
# Remove generated binaries and intermediate files
clean:
	rm -f $(TESTNAME) no-offload host-offload gpu-offload  obin *.i *.ii *.bc *.lk a.out-* *.ll *.s *.o *.cubin
