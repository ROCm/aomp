
TEST_NAME=vector_copy
CL_FILE=vector_copy

mkfile_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ifneq ($(CURDIR)/,$(mkfile_dir))
  _in_dir := $(mkfile_dir)
else
  _in_dir := 
endif
include $(mkfile_dir)../../Makefile.find_gpu_and_install_dir

CFLAGS = -std=c++11 -I$(LLVM_INSTALL_DIR)/include -I$(LLVM_INSTALL_DIR)/../include
LFLAGS = -L$(LLVM_INSTALL_DIR)/lib -L$(LLVM_INSTALL_DIR)/lib -lhsa-runtime64 -Wl,-rpath=$(LLVM_INSTALL_DIR)/lib
# find the current gpu, default to vega10 if unknown

CXX_FILES := $(wildcard $(_in_dir)*.cpp)
OBJ_FILES := $(addprefix obj/, $(notdir $(CXX_FILES:.cpp=.o)))

all: $(TEST_NAME) $(CL_FILE).hsaco

obj/%.o: $(_in_dir)%.cpp
	mkdir -p obj
	$(CXX) -c $(CFLAGS) -o $@ $<

$(TEST_NAME): $(OBJ_FILES) $(COMMON_OBJ_FILES)
	$(CXX) $(OBJ_FILES) $(LFLAGS) -o $(TEST_NAME)

$(CL_FILE).hsaco :
	$(LLVM_INSTALL_DIR)/bin/cloc.sh -cl12 -mcpu $(LLVM_GPU_ARCH) $(_in_dir)$(CL_FILE).cl -o $(CL_FILE).hsaco

clean:
	rm -rf obj/*o *.hsaco $(TEST_NAME)

run: $(TEST_NAME) $(CL_FILE).hsaco
	./$(TEST_NAME)


