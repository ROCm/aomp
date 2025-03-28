#-----------------------------------------------------------------------
#
#  Makefile:
#
#  Run "make help" to see other options for this Makefile

TESTNAME = print_device
TESTSRC  = print_device.c

mkfile_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ifneq ($(CURDIR)/,$(mkfile_dir))
  TESTSRC := $(mkfile_dir)$(TESTSRC)
endif
include $(mkfile_dir)../../inc/find_gpu_and_install_dir.mk

CC = $(LLVM_INSTALL_DIR)/bin/clang

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
	$(RUNENV) ROCR_VISIBLE_DEVICES=0,1 ./$(TESTNAME)
	$(RUNENV) $(LLVM_INSTALL_DIR)/bin/gpurun ./$(TESTNAME)

runmpi: $(TESTNAME)
	mpirun -np 2 $(RUNENV) $(LLVM_INSTALL_DIR)/bin/gpurun ./$(TESTNAME)

include $(mkfile_dir)../../inc/obin.mk
include $(mkfile_dir)../../inc/help.mk

# Cleanup anything this makefile can create
clean:
	rm -f $(TESTNAME) obin *.i *.ii *.bc *.lk a.out-* *.ll *.s *.o *.cubin
