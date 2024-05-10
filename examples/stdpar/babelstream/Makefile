TESTSRC1 = main.cpp
TESTSRC2 = STDDataStream.cpp
TESTSRC3 = STDIndicesStream.cpp

mkfile_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ifneq ($(CURDIR)/,$(mkfile_dir))
  # Prefix source files if Makefile is not in current directory
  TESTSRC1 := $(mkfile_dir)$(TESTSRC1)
  TESTSRC2 := $(mkfile_dir)$(TESTSRC2)
  TESTSRC3 := $(mkfile_dir)$(TESTSRC3)
endif

# Set LLVM_INSTALL_DIR and LLVM_GPU_ARCH
include $(mkfile_dir)../../inc/find_gpu_and_install_dir.mk

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
std-data-stream: main.o STDDataStream.o 
	$(CC) -DSTD_DATA $(opts) $^ -o $@
main.o : $(TESTSRC1)
	$(CC) -DSTD_DATA -c $(opts) $^ -o $@
STDDataStream.o : $(TESTSRC2)
	$(CC) -DSTD_DATA -c $(opts) $^ -o $@

std-ind-stream: i_main.o STDIndicesStream.o 
	$(CC) -DSTD_INDICES $(opts) $^ -o $@
i_main.o : $(TESTSRC1)
	$(CC) -DSTD_INDICES -c $(opts) $^ -o $@
STDIndicesStream.o : $(TESTSRC3)
	$(CC) -DSTD_INDICES -c $(opts) $^ -o $@

# stdpar requires unified shared memory so set HSA_XNACK=1
run: std-ind-stream std-data-stream
	HSA_XNACK=1 $(LLVM_INSTALL_DIR)/bin/gpurun ./std-data-stream
	HSA_XNACK=1 $(LLVM_INSTALL_DIR)/bin/gpurun ./std-ind-stream

clean:
	@[ -f ./std-data-stream ] && rm ./std-data-stream; true
	@[ -f ./main.o ] && rm ./main.o; true
	@[ -f ./STDDataStream.o ] && rm ./STDDataStream.o; true
	@[ -f ./std-ind-stream ] && rm ./std-ind-stream; true
	@[ -f ./i_main.o ] && rm ./i_main.o; true
	@[ -f ./STDIndicesStream.o ] && rm ./STDIndicesStream.o; true
