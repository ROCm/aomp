INC_PATH  ?= $(AOMP)/include
LIB_PATH  ?= $(AOMP)/roctracer/lib
ROCM_PATH ?= $(AOMP)
HIP_VDI   ?= 0
ITERS     ?= 100
HIP_PATH  ?= $(AOMP)

ROC_LIBS  = -Wl,--rpath,${LIB_PATH} $(LIB_PATH)/libroctracer64.so $(LIB_PATH)/libroctx64.so

HIPCC=$(HIP_PATH)/bin/hipcc

EXECUTABLE=./MatrixTranspose
OBJECTS = MatrixTranspose.o
FLAGS =-g -fPIC $(INC_PATH:%=-I%) -I$(ROCM_PATH)/include/roctracer -I$(ROCM_PATH)/include/hip -I$(ROCM_PATH)/include -DLOCAL_BUILD=1 -DHIP_VDI=${HIP_VDI} -DITERATIONS=$(ITERS) -DAMD_INTERNAL_BUILD=1

ifeq ($(C_TEST), 1)
	COMP=${CC}
	SOURCES = MatrixTranspose.c
	FLAGS += -DHIP_TEST=0 -D__HIP_PLATFORM_HCC__=1
else
	COMP=$(HIPCC)
	SOURCES = MatrixTranspose.cpp
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

