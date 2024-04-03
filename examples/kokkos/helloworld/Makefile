#-----------------------------------------------------------------------
#
#  Makefile: KOKKOS Makefile for amdgcn target.
#            make run
#
#  Run "make help" to see other options for this Makefile

TESTNAME = helloworld
TESTSRC = $(TESTNAME).cpp

mkfile_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ifneq ($(CURDIR)/,$(mkfile_dir))
  TESTSRC := $(mkfile_dir)$(TESTSRC)
endif
include $(mkfile_dir)../../Makefile.find_gpu_and_install_dir

CC          = $(LLVM_INSTALL_DIR)/bin/clang++
LLVM_GPU_ARCHTARGET ?= amdgcn-amd-amdhsa
KOKKOS_INSTALL_PREFIX ?= $(HOME)

ifeq ($(KOKKOS_USE_HIP),1)
  is_hip    = hip
  KOKKOS_ROOT = $(KOKKOS_INSTALL_PREFIX)/kokkos_hip.$(LLVM_GPU_ARCH)
else
  is_hip    =
  KOKKOS_ROOT = $(KOKKOS_INSTALL_PREFIX)/kokkos_omp.$(LLVM_GPU_ARCH)
endif

INCLUDES     = -I$(KOKKOS_ROOT)/include
TARGET_FLAGS = -fopenmp --offload-arch=$(LLVM_GPU_ARCH)$(AOMP_TARGET_FEATURES)
CPPFLAGS     = $(INCLUDES) -Wall -Wextra -O3 -DNDEBUG  -w $(TARGET_FLAGS) -std=c++17 -DGTEST_HAS_PTHREAD=0 -DGTEST_HAS_TR1_TUPLE=0 -DKOKKOS_DEPENDENCE -DKOKKOS_WORKAROUND_OPENMPTARGET_CLANG
LINKFLAGS    = $(enableopt) -Wall -Wextra -O3 -DNDEBUG -w $(TARGET_FLAGS)

# Archive may be at lib64 for non-debian systems.
# Ubuntu 20.04 has a /usr/lib64 but Kokkos does not use the lib64 directory.
# Add a check for debian on os-release.
DEBIAN := $(shell cat /etc/os-release | grep debian)
ifneq (,$(wildcard /usr/lib64))
  ifeq (,$(DEBIAN))
    KOKKOS_LIB   = $(KOKKOS_ROOT)/lib64/libkokkoscore.a
  else
    KOKKOS_LIB   = $(KOKKOS_ROOT)/lib/libkokkoscore.a
  endif
else
  KOKKOS_LIB   = $(KOKKOS_ROOT)/lib/libkokkoscore.a
endif

ifeq ($(OFFLOAD_DEBUG),1)
  $(info    DEBUG Mode ON)
  CCENV  = env LIBRARY_PATH=$(LLVM_INSTALL_DIR)/lib-debug
  RUNENV = LIBOMPTARGET_DEBUG=1
endif

ifeq ($(VERBOSE),1)
  $(info    Compilation VERBOSE Mode ON)
  CPPFLAGS += -v
endif

ifeq ($(TEMPS),1)
  $(info    Compilation and linking save-temp Mode ON)
  CPPFLAGS += -save-temps
endif

CPPFLAGS += $(EXTRA_CPPFLAGS)

$(TESTNAME): $(TESTNAME).o $(KOKKOS_LIB)
	$(CCENV) $(CC) $(LINKFLAGS) -o $@ $(TESTNAME).o $(KOKKOS_LIB) -ldl

$(TESTNAME).o: $(TESTSRC) $(KOKKOS_LIB)
	$(CCENV) $(CC) $(CPPFLAGS) -o $@ -c $(TESTSRC)

run: $(TESTNAME)
	$(RUNENV) ./$(TESTNAME)

$(KOKKOS_LIB):
	$(LLVM_INSTALL_DIR)/bin/kokkos_build.sh $(is_hip)

help:
	@echo
	@echo "Source[s]:		$(TESTSRC)"
	@echo "Application binary:    	$(TESTNAME)"
	@echo "Target GPU:		$(LLVM_GPU_ARCH)"
	@echo "Compiler: 		$(CC)"
	@echo "Compile flags:		$(CPPFLAGS)"
	@echo "KOKKOS install prefix:	$(KOKKOS_INSTALL_PREFIX)"
ifeq (sm_,$(findstring sm_,$(LLVM_GPU_ARCH)))
	@echo "CUDA installation:	$(CUDA)"
endif
	@echo
	@echo "This Makefile supports these targets:"
	@echo
	@echo " make			 // Builds $(TESTNAME) and KOKKOS "
	@echo " make run		 // Executes $(TESTNAME) "
	@echo
	@echo " make clean              // Deletes test build  "
	@echo " make clean_kokkos       // Deletes KOKKOS build "
	@echo " make help"
	@echo
	@echo "Environment variables used by this Makefile:"
	@echo "  LLVM_INSTALL_DIR=<dir>"
	@echo "  EXTRA_CFLAGS=<args>  extra arguments for compiler"
	@echo "  OFFLOAD_DEBUG=n      if n=1, compile and run in Debug mode"
	@echo "  VERBOSE=n            if n=1, add verbose output"
	@echo "  TEMPS=1              do not delete intermediate files"
ifeq (sm_,$(findstring sm_,$(LLVM_GPU_ARCH)))
	@echo "  CUDA=<dir>           CUDA install dir, default=/usr/local/cuda"
endif
	@echo

# Cleanup anything this makefile can create
clean:
	rm -rf $(TESTNAME) obin *.i *.ii *.bc *.lk a.out-* *.ll *.s *.o *.cubin

clean_kokkos:
	$(LLVM_INSTALL_DIR)/bin/kokkos_build.sh clean
