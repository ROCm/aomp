
mkfile_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ifneq ($(CURDIR)/,$(mkfile_dir))
  _in_dir := $(mkfile_dir)
else
  _in_dir :=
endif

mkfile_find_loc := $(mkfile_dir)../../../Makefile.find_gpu_and_install_dir
ifeq ("$(wildcard $(mkfile_find_loc))","")
  mkfile_find_loc := $(mkfile_dir)../Makefile.find_gpu_and_install_dir
endif
include $(mkfile_find_loc)

INC_PATH  ?= $(LLVM_INSTALL_DIR)/include
LIB_PATH  ?= $(LLVM_INSTALL_DIR)/lib/roctracer
ROCM_PATH ?= $(HIPDIR)
HIP_VDI   ?= 0
ITERS     ?= 100
HIP_PATH  ?= $(HIPDIR)

ifeq ("$(wildcard $(LIB_PATH)/libroctracer_tool.so)","")
  # These changes support ROCm installed compiler, LLVM_INSTALL_DIR=/opt/rocm/lib/llvm
  LIB_PATH := $(LLVM_INSTALL_DIR)/../roctracer/lib
  ROCM_PATH := $(LLVM_INSTALL_DIR)
endif

ROC_LIBS  = -Wl,--rpath,${LIB_PATH} $(HIPDIR)/lib/libroctracer64.so $(HIPDIR)/lib/libroctx64.so

EXECUTABLE=./MatrixTranspose
OBJECTS = MatrixTranspose.o
FLAGS =-g -fPIC $(INC_PATH:%=-I%) -I$(HIPDIR)/include/roctracer -I$(ROCM_PATH)/include/hip -I$(ROCM_PATH)/include -DLOCAL_BUILD=1 -DHIP_VDI=${HIP_VDI} -DITERATIONS=$(ITERS) -DAMD_INTERNAL_BUILD=1

ifeq ($(C_TEST), 1)
	COMP=${CC}
	SOURCES = $(_in_dir)MatrixTranspose.c
	FLAGS += -DHIP_TEST=0 -D__HIP_PLATFORM_AMD__=1
else
	COMP=$(HIPCC).bin
	SOURCES = $(_in_dir)MatrixTranspose.cpp
	FLAGS += -DHIP_TEST=1 --rocm-path=$(ROCM_PATH) -D__HIP_PLATFORM_AMD__=1 --offload-arch=$(LLVM_GPU_ARCH) -Wno-unused-result
endif
ifeq ($(MGPU_TEST), 1)
	FLAGS += -DMGPU_TEST=1
endif
ifeq ($(HIP_API_ACTIVITY_ON), 1)
        FLAGS += -DHIP_API_ACTIVITY_ON=1
endif

.PHONY: test

all: clean $(EXECUTABLE)

$(OBJECTS): $(SOURCES)
	$(COMP) $(FLAGS) -c -o $@ $<

$(EXECUTABLE): $(OBJECTS)
	env HIP_PLATFORM=amd $(HIPCC) $(OBJECTS) -o $@ $(ROC_LIBS)

test: $(EXECUTABLE)
	LD_PRELOAD="librocprofiler64.so" $(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)
	rm -f $(HIP_PATH)/src/*.o

