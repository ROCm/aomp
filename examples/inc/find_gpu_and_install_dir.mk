#-----------------------------------------------------------------------
#
#  find_gpu_and_install_dir.mk
#     Include for examples to find LLVM installation and GPU arch.
#     This include
#        - sets LLVM_INSTALL_DIR, LLVM_GPU_TRIPLE, and LLVM_GPU_ARCH
#        - sets CUDA and LFLAGS to find CUDA lib if NVIDIA gpu
#          CUDA and LFLAGS to find CUDA lib if NVIDIA gpu
#        - HIPDIR and HIPCC=$(HIPDIR)/bin/hipcc
 
# NOTE: THIS FIRST BLOCK CAN EVENTUALLY BE DELETED
# If LLVM_INSTALL_DIR not preset and AOMP is, then use AOMP and print deprecation warning
ifeq ("$(wildcard $(LLVM_INSTALL_DIR))","")
ifneq ("$(wildcard $(AOMP))","")
  $(info 'WARNING: Either preset LLVM_INSTALL_DIR or unset AOMP. Using LLVM_INSTALL_DIR=$(AOMP).')
  LLVM_INSTALL_DIR = $(AOMP)
endif
endif

# If LLVM_INSTALL_DIR not preset, seach rocm at /opt/rocm/lib/llvm, and then ~/rocm/aomp
ifeq ("$(wildcard $(LLVM_INSTALL_DIR))","")
  ifneq ($(LLVM_INSTALL_DIR),)
    $(warning Specified LLVM_INSTALL_DIR:$(LLVM_INSTALL_DIR) NOT FOUND)
  endif
  this_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
  # this_dir ... finds the compiler if example Make is run directly from the installation e.g.:
  #   make -f /opt/rocm/share/openmp-extras/examples/openmp/veccopy/Makefile run
  LLVM_INSTALL_SEARCH_DIRS ?= /opt/rocm/lib/llvm $(this_dir)../../../llvm $(HOME)/rocm/aomp
  # Select the first directory found from above list.
  LLVM_INSTALL_DIR := $(word 1, $(strip $(foreach dir,$(LLVM_INSTALL_SEARCH_DIRS),$(wildcard $(dir)))))
  ifeq ("$(wildcard $(LLVM_INSTALL_DIR))","")
    $(error Please install ROCm, AOMP, or set LLVM_INSTALL_DIR to LLVM install dir)
  endif
endif

# Test for preset GPUs
ifeq ($(LLVM_GPU_ARCH),)
  # If not preset, get offload arch from either nvidia-arch or amdgpu-arch
  _amdgpu_mod_dir = $(shell ls -d /sys/module/amdgpu 2>/dev/null)
  ifeq ("$(wildcard $(_amdgpu_mod_dir))","")
    LLVM_GPU_ARCH = $(shell $(LLVM_INSTALL_DIR)/bin/nvptx-arch | head -n 1 )
  else
    LLVM_GPU_ARCH = $(shell $(LLVM_INSTALL_DIR)/bin/amdgpu-arch | head -n 1 )
  endif
endif

ifeq ($(strip $(LLVM_GPU_ARCH)),)
  $(error Could NOT detect a GPU to set LLVM_GPU_ARCH! To test compile only, set LLVM_GPU_ARCH=gfx90a)
endif

ifeq (sm_,$(findstring sm_,$(LLVM_GPU_ARCH)))
  LLVM_GPU_TRIPLE = nvptx64-nvidia-cuda
else
  LLVM_GPU_TRIPLE = amdgcn-amd-amdhsa
endif

# Find where HIP is installed and set HIPDIR and HIPCC
HIPDIR ?= $(LLVM_INSTALL_DIR)
ifeq ("$(wildcard $(HIPDIR)/bin/hipcc)","")
  HIPDIR := $(LLVM_INSTALL_DIR)/..
  ifeq ("$(wildcard $(HIPDIR)/bin/hipcc)","")
    HIPDIR := $(LLVM_INSTALL_DIR)/../..
     ifeq ("$(wildcard $(HIPDIR)/bin/hipcc)","")
	HIPDIR := /opt/rocm
     endif
  endif
endif
HIPCC := $(HIPDIR)/bin/hipcc
