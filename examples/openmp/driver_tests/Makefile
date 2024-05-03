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

TESTNAME = driver_tests
TESTSRC  = $(TESTNAME).c

mkfile_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ifneq ($(CURDIR)/,$(mkfile_dir))
  TESTSRC := $(mkfile_dir)$(TESTSRC)
endif
include $(mkfile_dir)../../inc/find_gpu_and_install_dir.mk

CC = $(LLVM_INSTALL_DIR)/bin/clang

CFLAGS := -fopenmp
OAFLAG := --offload-arch=$(LLVM_GPU_ARCH)

ifeq ($(VERBOSE),1)
  $(info    Compilation VERBOSE Mode ON)
  CFLAGS += -v
endif

CFLAGS += $(EXTRA_CFLAGS)

# ----- Demo compile and link in one step, no object code saved
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

run: $(TESTNAME)
	./$(TESTNAME)

# Cleanup anything this makefile can create
clean:
	rm -f $(TESTNAME) obin *.i *.ii *.bc *.lk a.out-* *.ll *.s *.o *.cubin
