
TEST_NAME=vector_copy
CU_FILE=vector_copy

UNAMEP = $(shell uname -p)
AOMP_CPUTARGET = $(UNAMEP)-pc-linux-gnu
ifeq ($(UNAMEP),ppc64le)
  AOMP_CPUTARGET = ppc64le-linux-gnu
endif
ifeq ($(AOMP),)
  AOMP = /usr/lib/aomp
else
  $(info AOMP environment variable using $(AOMP))
endif
# find the current gpu, default to vega10 if unknown
AOMP_GPU       ?= $(shell $(AOMP)/bin/mygpu -d gfx900)

ifeq (sm_,$(findstring sm_,$(AOMP_GPU)))
  AOMP_GPUTARGET = nvptx64-nvidia-cuda
else
  AOMP_GPUTARGET = amdgcn-amd-amdhsa
endif

#LFLAGS= -g  -Wl,--unresolved-symbols=ignore-in-shared-libs
LFLAGS = 
CFLAGS =  -std=c++11 -D__HIP_PLATFORM_HCC__ -O3 -target $(AOMP_CPUTARGET) -fopenmp -fopenmp-targets=$(AOMP_GPUTARGET) -Xopenmp-target=$(AOMP_GPUTARGET) -march=$(AOMP_GPU)
FUNOPTS = -vv -ll
FUNOPTS = 

ifeq ($(OFFLOAD_DEBUG),1)
  $(info    DEBUG Mode ON)
  CCENV  = env LIBRARY_PATH=$(AOMP)/lib-debug
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

ifeq (sm_,$(findstring sm_,$(AOMP_GPU)))
  CUDA   ?= /usr/local/cuda
  LFLAGS += -L$(CUDA)/targets/$(UNAMEP)-linux/lib -lcudart
endif

CFLAGS += $(EXTRA_CFLAGS)

CXX_FILES := $(wildcard *.cpp)
OBJ_FILES := $(addprefix obj/, $(notdir $(CXX_FILES:.cpp=.o)))

all: $(TEST_NAME) $(CU_FILE).hsaco

$(TEST_NAME): $(OBJ_FILES) $(COMMON_OBJ_FILES)
	$(AOMP)/bin/clang++  $(OBJ_FILES) $(CFLAGS) -L$(AOMP)/lib -lamdhip64 -o $(TEST_NAME) $(LFLAGS)

$(CU_FILE).hsaco :
	$(AOMP)/bin/cloc.sh $(FUNOPTS)  $(CU_FILE).cu

obj/%.o: %.cpp
	mkdir -p obj
	$(AOMP)/bin/clang++ -I$(AOMP)/include -c $(CFLAGS) -o $@ $<

clean:
	rm -rf obj/*o *.hsaco *.ll $(TEST_NAME)


run:
	./$(TEST_NAME)


