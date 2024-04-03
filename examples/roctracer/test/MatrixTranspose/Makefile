

mkfile_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ifneq ($(CURDIR)/,$(mkfile_dir))
  _in_dir := $(mkfile_dir)
else
  _in_dir :=
endif
include $(mkfile_dir)../../../Makefile.find_gpu_and_install_dir

INC_PATH  ?= $(LLVM_INSTALL_DIR)/include
LIB_PATH  ?= $(LLVM_INSTALL_DIR)/lib
ROCM_PATH ?= $(HIPDIR)
HIP_VDI   ?= 0
ITERS     ?= 100
HIP_PATH  ?= $(HIPDIR)

ifeq ("$(wildcard $(LIB_PATH)/libroctracer64.so)","")
  LIB_PATH := $(LLVM_INSTALL_DIR)/../roctracer/lib
endif

ROC_LIBS  = -Wl,--rpath,${LIB_PATH} $(LIB_PATH)/libroctracer64.so $(LIB_PATH)/libroctx64.so

EXECUTABLE=./MatrixTranspose
OBJECTS = MatrixTranspose.o
FLAGS =-g $(INC_PATH:%=-I%) -I$(ROCM_PATH)/include/roctracer -I$(ROCM_PATH)/include/hip -I$(ROCM_PATH)/include -DLOCAL_BUILD=1 -DHIP_VDI=${HIP_VDI} -DITERATIONS=$(ITERS) -DAMD_INTERNAL_BUILD=1

ifeq ($(C_TEST), 1)
	COMP=${CC}
	SOURCES = $(_in_dir)MatrixTranspose.c
	FLAGS += -DHIP_TEST=0 -D__HIP_PLATFORM_HCC__=1
else
	COMP=$(HIPCC)
	SOURCES = $(_in_dir)MatrixTranspose.cpp
	FLAGS += -DHIP_TEST=1 --rocm-path=$(ROCM_PATH)
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
	$(HIPCC) $(OBJECTS) -o $@ $(ROC_LIBS)

test: $(EXECUTABLE)
	LD_PRELOAD="librocprofiler64.so" $(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)
	rm -f $(HIP_PATH)/src/*.o

