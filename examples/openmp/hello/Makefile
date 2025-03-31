#-----------------------------------------------------------------------
# Makefile for building and running OpenMP Hello World examples.
# Demonstrates different modes of execution using OpenMP.
#-----------------------------------------------------------------------

# Define test names and source files
TESTNAME1 = hello1
TESTSRC1  = hello1.c

TESTNAME2 = hello2
TESTSRC2  = hello2.c

TESTNAME3 = hello3
TESTSRC3  = hello3.c

TESTNAME4 = hello4
TESTSRC4  = hello4.c

TESTNAME5 = hello5
TESTSRC5  = hello5.c

# Determine the directory of the Makefile
mkfile_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Adjust source file paths if running out-of-tree
ifneq ($(CURDIR)/,$(mkfile_dir))
  TESTSRC1 := $(mkfile_dir)$(TESTSRC1)
endif
ifneq ($(CURDIR)/,$(mkfile_dir))
  TESTSRC2 := $(mkfile_dir)$(TESTSRC2)
endif
ifneq ($(CURDIR)/,$(mkfile_dir))
  TESTSRC3 := $(mkfile_dir)$(TESTSRC3)
endif
ifneq ($(CURDIR)/,$(mkfile_dir))
  TESTSRC4 := $(mkfile_dir)$(TESTSRC4)
endif
ifneq ($(CURDIR)/,$(mkfile_dir))
  TESTSRC5 := $(mkfile_dir)$(TESTSRC5)
endif

# Include helper Makefile for GPU and compiler setup
include $(mkfile_dir)../../inc/find_gpu_and_install_dir.mk

# Compiler and flags
CC       = $(LLVM_INSTALL_DIR)/bin/clang
CFLAGS1   = -O3 -fopenmp --offload-arch=$(LLVM_GPU_ARCH)$(AOMP_TARGET_FEATURES)
CFLAGS2   = -O3 -fopenmp --offload-arch=$(LLVM_GPU_ARCH)$(AOMP_TARGET_FEATURES)
CFLAGS3   = -O3 -fopenmp --offload-arch=$(LLVM_GPU_ARCH)$(AOMP_TARGET_FEATURES)
CFLAGS4   = -O3 -fopenmp --offload-arch=$(LLVM_GPU_ARCH)$(AOMP_TARGET_FEATURES)
CFLAGS5   = -O3 -fopenmp --offload-arch=$(LLVM_GPU_ARCH)$(AOMP_TARGET_FEATURES)

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

# Enable saving temporary files if TEMPS is set
ifeq ($(TEMPS),1)
  $(info    Compilation and linking save-temp Mode ON)
  CFLAGS += -save-temps 
endif

CFLAGS += $(EXTRA_CFLAGS)

# Build all examples
all: $(TESTNAME1) $(TESTNAME2) $(TESTNAME3) $(TESTNAME4) $(TESTNAME5) 

# Compile and link each example
$(TESTNAME1): $(TESTSRC1)
	$(CCENV) $(CC) $(CFLAGS1) $(LFLAGS) $^ -o $@

$(TESTNAME2): $(TESTSRC2)
	$(CCENV) $(CC) $(CFLAGS2) $(LFLAGS) $^ -o $@

$(TESTNAME3): $(TESTSRC3)
	$(CCENV) $(CC) $(CFLAGS3) $(LFLAGS) $^ -o $@

$(TESTNAME4): $(TESTSRC4)
	$(CCENV) $(CC) $(CFLAGS4) $(LFLAGS) $^ -o $@

$(TESTNAME5): $(TESTSRC5)
	$(CCENV) $(CC) $(CFLAGS5) $(LFLAGS) $^ -o $@

# Run all examples
run: $(TESTNAME1) $(TESTNAME2) $(TESTNAME3) $(TESTNAME4) $(TESTNAME5)
	$(RUNENV) ./$(TESTNAME1)
	$(RUNENV) ./$(TESTNAME2)
	$(RUNENV) ./$(TESTNAME3)
	$(RUNENV) ./$(TESTNAME4)
	$(RUNENV) ./$(TESTNAME5)

# Include additional helper Makefiles
include $(mkfile_dir)../../inc/obin.mk
include $(mkfile_dir)../../inc/help.mk

# Cleanup generated files
clean:
	rm -f $(TESTNAME1) $(TESTNAME2) $(TESTNAME3) $(TESTNAME4) $(TESTNAME5) \
		obin *.i *.ii *.bc *.lk a.out-* *.ll *.s *.o *.cubin
