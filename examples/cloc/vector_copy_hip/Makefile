
TEST_NAME=vector_copy
HIP_FILE=vector_copy

include ../../Makefile.find_gpu_and_install_dir

CFLAGS = -c -std=c++11 -D__HIP_PLATFORM_AMD__ -I$(LLM_INSTALL_DIR)/../include -I$(LLVM_INSTALL_DIR)/include
LFLAGS = -L$(LLVM_INSTALL_DIR)/lib -L$(LLVM_INSTALL_DIR)/../lib -lamdhip64 -Wl,-rpath,$(LLVM_INSTALL_DIR)/lib -Wl,-rpath,$(LLVM_INSTALL_DIR)/../lib

CXX_FILES := $(wildcard *.cpp)
OBJ_FILES := $(addprefix obj/, $(notdir $(CXX_FILES:.cpp=.o)))

all: $(TEST_NAME) $(HIP_FILE).hsaco

$(TEST_NAME): $(OBJ_FILES) $(COMMON_OBJ_FILES)
	$(LLVM_INSTALL_DIR)/bin/clang++ $(OBJ_FILES) $(LFLAGS) -o $(TEST_NAME)

$(HIP_FILE).hsaco:
	$(LLVM_INSTALL_DIR)/bin/cloc.sh -mcpu $(LLVM_GPU_ARCH) $(HIP_FILE).cu

obj/%.o: %.cpp
	mkdir -p obj
	$(LLVM_INSTALL_DIR)/bin/clang++ $(CFLAGS) -o $@ $<

clean:
	rm -rf obj/*o *.hsaco *.ll $(TEST_NAME)

run:
	./$(TEST_NAME)
