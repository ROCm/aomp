
TEST_NAME=vector_copy
CU_FILE=vector_copy

include ../../Makefile.find_gpu_and_install_dir

ifeq (sm_,$(findstring sm_,$(LLVM_GPU_ARCH)))
  LLVM_GPU_ARCHTARGET = nvptx64-nvidia-cuda
else
  LLVM_GPU_ARCHTARGET = amdgcn-amd-amdhsa
endif

#LFLAGS= -g  -Wl,--unresolved-symbols=ignore-in-shared-libs
LFLAGS = 
CFLAGS =  -std=c++11 -D__HIP_PLATFORM_AMD__ -O3 -fopenmp --offload-arch=$(LLVM_GPU_ARCH)$(AOMP_TARGET_FEATURES)
FUNOPTS = -vv -ll
FUNOPTS = 

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

CXX_FILES := $(wildcard *.cpp)
OBJ_FILES := $(addprefix obj/, $(notdir $(CXX_FILES:.cpp=.o)))

all: $(TEST_NAME) $(CU_FILE).hsaco

$(TEST_NAME): $(OBJ_FILES) $(COMMON_OBJ_FILES)
	$(LLVM_INSTALL_DIR)/bin/clang++  $(OBJ_FILES) $(CFLAGS) -L$(HIPDIR)/lib -lamdhip64 -o $(TEST_NAME) $(LFLAGS)

$(CU_FILE).hsaco :
	$(HIPDIR)/bin/cloc.sh $(FUNOPTS) -mcpu $(LLVM_GPU_ARCH) $(CU_FILE).cu

obj/%.o: %.cpp
	mkdir -p obj
	$(LLVM_INSTALL_DIR)/bin/clang++ -I$(LLVM_INSTALL_DIR)/include -c $(CFLAGS) -o $@ $<

clean:
	rm -rf obj/*o *.hsaco *.ll $(TEST_NAME)


run:
	./$(TEST_NAME)


