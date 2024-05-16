#-----------------------------------------------------------------------
#
#  Makefile: KOKKOS Makefile for amdgcn target.
#            make run
#
#  Run "make help" to see other options for this Makefile

mkfile_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
include $(mkfile_dir)../../inc/find_gpu_and_install_dir.mk

TESTNAME = helloworld
TESTSRC = $(TESTNAME).cpp

ifneq ($(CURDIR)/,$(mkfile_dir))
  TESTSRC := $(mkfile_dir)$(TESTSRC)
endif

CC := $(LLVM_INSTALL_DIR)/bin/clang++
KOKKOS_INSTALL_PREFIX ?= /work/$(USER)

ifeq ($(KOKKOS_USE_HIP),1)
  is_hip    = hip
  KOKKOS_INSTALL_DIR = $(KOKKOS_INSTALL_PREFIX)/kokkos-3.7.01_hip.$(LLVM_GPU_ARCH)
else
  is_hip    =
  KOKKOS_INSTALL_DIR = $(KOKKOS_INSTALL_PREFIX)/kokkos-3.7.01_omp.$(LLVM_GPU_ARCH)
endif

INCLUDES     = -I$(KOKKOS_INSTALL_DIR)/include
TARGET_FLAGS = -fopenmp --offload-arch=$(LLVM_GPU_ARCH)$(AOMP_TARGET_FEATURES)
CPPFLAGS     = $(INCLUDES) -Wall -Wextra -O3 -DNDEBUG  -w $(TARGET_FLAGS) -std=c++17 -DGTEST_HAS_PTHREAD=0 -DGTEST_HAS_TR1_TUPLE=0 -DKOKKOS_DEPENDENCE -DKOKKOS_WORKAROUND_OPENMPTARGET_CLANG
LINKFLAGS    = $(enableopt) -Wall -Wextra -O3 -DNDEBUG -w $(TARGET_FLAGS)

# Archive may be at lib64 for non-debian systems.
# Ubuntu 20.04 has a /usr/lib64 but Kokkos does not use the lib64 directory.
# Add a check for debian on os-release.
DEBIAN := $(shell cat /etc/os-release | grep debian)
ifneq (,$(wildcard /usr/lib64))
  ifeq (,$(DEBIAN))
    KOKKOS_LIB   = $(KOKKOS_INSTALL_DIR)/lib64/libkokkoscore.a
  else
    KOKKOS_LIB   = $(KOKKOS_INSTALL_DIR)/lib/libkokkoscore.a
  endif
else
  KOKKOS_LIB   = $(KOKKOS_INSTALL_DIR)/lib/libkokkoscore.a
endif

CPPFLAGS += $(EXTRA_CPPFLAGS)

$(TESTNAME): $(TESTNAME).o $(KOKKOS_LIB)
	$(CCENV) $(CC) $(LINKFLAGS) -o $@ $(TESTNAME).o $(KOKKOS_LIB) -ldl

$(TESTNAME).o: $(TESTSRC) $(KOKKOS_LIB)
	$(CCENV) $(CC) $(CPPFLAGS) -o $@ -c $(TESTSRC)

run: $(TESTNAME)
	$(RUNENV) ./$(TESTNAME)

$(KOKKOS_LIB):
	env AOMP=$(LLVM_INSTALL_DIR) AOMP_INSTALL_DIR=$(LLVM_INSTALL_DIR) AOMP_GPU=$(LLVM_GPU_ARCH) KOKKOS_BUILD_PREFIX=$(KOKKOS_INSTALL_PREFIX) KOKKOS_INSTALL_DIR=$(KOKKOS_INSTALL_DIR) $(LLVM_INSTALL_DIR)/bin/kokkos_build.sh $(is_hip)

help:
	@echo
	@echo "Source:			$(TESTSRC)"
	@echo "Application binary:    	$(TESTNAME)"
	@echo "KOKKOS lib:		$(KOKKOS_LIB)"
	@echo "Target GPU:		$(LLVM_GPU_ARCH)"
	@echo "Compiler: 		$(CC)"
	@echo "KOKKOS build script:	$(LLVM_INSTALL_DIR)/bin/kokkos_build.sh"
	@echo "KOKKOS install dir:	$(KOKKOS_INSTALL_DIR)"
	@echo "Compile flags:		$(CPPFLAGS)"

ifeq (sm_,$(findstring sm_,$(LLVM_GPU_ARCH)))
	@echo "CUDA installation:	$(CUDA)"
endif
	@echo
	@echo "This Makefile supports these targets:"
	@echo
	@echo " make			 // Builds $(TESTNAME) and KOKKOS "
	@echo " make run		 // Executes $(TESTNAME) "
	@echo
	@echo " make clean              // Deletes $(TESTNAME) but not kokkos"
	@echo " make clean_kokkos       // Deletes KOKKOS build "
	@echo " make help"
	@echo
	@echo "Environment variables used by this Makefile:"
	@echo "  LLVM_INSTALL_DIR	$(LLVM_INSTALL_DIR)"
	@echo "  LLVM_GPU_ARCH		$(LLVM_GPU_ARCH)" 
	@echo "  KOKKOS_INSTALL_PREFIX	$(KOKKOS_INSTALL_PREFIX)" 
	@echo "  EXTRA_CFLAGS		extra arguments for compiler"
ifeq (sm_,$(findstring sm_,$(LLVM_GPU_ARCH)))
	@echo "  CUDA=<dir>           CUDA install dir, default=/usr/local/cuda"
endif
	@echo

# Cleanup anything this makefile can create
clean:
	rm -rf $(TESTNAME) obin *.i *.ii *.bc *.lk a.out-* *.ll *.s *.o *.cubin

clean_kokkos:
	$(LLVM_INSTALL_DIR)/bin/kokkos_build.sh clean
