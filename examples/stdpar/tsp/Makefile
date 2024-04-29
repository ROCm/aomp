TESTSRC1 = tsp.cpp

mkfile_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ifneq ($(CURDIR)/,$(mkfile_dir))
  # Prefix source files if Makefile is not in current directory
  TESTSRC1 := $(mkfile_dir)$(TESTSRC1)
endif

# Set LLVM_INSTALL_DIR and LLVM_GPU_ARCH
include $(mkfile_dir)../../Makefile.find_gpu_and_install_dir

HIPSTDPARPATH = /opt/rocm/include/thrust/system/hip
ifeq ("$(wildcard $(HIPSTDPARPATH))","")
  $(error "ERROR: Missing dir $(HIPSTDPARPATH). stdpar requires installation of ROCm thrust")
endif
HIPSTDPARTHRUSTPATH = /opt/rocm/include
ifeq ("$(wildcard $(HIPSTDPARTHRUSTPATH)/thrust)","")
  $(error "ERROR: Missing dir $(HIPSTDPARTHRUSTPATH)/thrust. stdpar requires installation of ROCm thrust")
endif
HIPSTDPARPRIMPATH = /opt/rocm/include
ifeq ("$(wildcard $(HIPSTDPARPRIMPATH)/rocprim)","")
  $(error "ERROR: Missing dir $(HIPSTDPARPRIMPATH)/rocprim. stdpar requires installation of ROCm prim")
endif

CC = $(LLVM_INSTALL_DIR)/bin/clang++
opts=-I. -DNDEBUG -O3 --hipstdpar --hipstdpar-path=$(HIPSTDPARPATH) --hipstdpar-thrust-path=$(HIPSTDPARTHRUSTPATH) --hipstdpar-prim-path=$(HIPSTDPARPRIMPATH) --offload-arch=$(LLVM_GPU_ARCH) -std=c++17

# Build both babelstream stdpar binaries
tsp:tsp.o
	$(CC) $(opts) $^ -o $@
tsp.o : $(TESTSRC1)
	$(CC) -c $(opts) $^ -o $@

# stdpar requires unified shared memory so set HSA_XNACK=1
run: tsp
	HSA_XNACK=1 $(LLVM_INSTALL_DIR)/bin/gpurun ./tsp

clean:
	@[ -f ./tsp ] && rm ./tsp; true
	@[ -f ./tsp.o ] && rm ./tsp.o; true
