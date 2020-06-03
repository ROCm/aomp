
TEST_NAME=vector_copy
HIP_FILE=vector_copy

ifeq ($(AOMP),)
  AOMP = /usr/lib/aomp
endif

CFLAGS = -c -std=c++11 -D__HIP_PLATFORM_HCC__ -I$(AOMP)/../include -I$(AOMP)/include
# Set the rpath to pick up AOMP libs in case rocm was installed. Also support looking in /opt/rocm/lib as AOMP in ROCm no longer builds hip.
LFLAGS = -L$(AOMP)/lib -L$(AOMP)/../lib -lhip_hcc -Wl,-rpath,$(AOMP)/lib -Wl,-rpath,$(AOMP)/../lib
# find the current gpu, default to vega10 if unknown
MYGPU = $(shell $(AOMP)/bin/mygpu -d gfx900)

CXX_FILES := $(wildcard *.cpp)
OBJ_FILES := $(addprefix obj/, $(notdir $(CXX_FILES:.cpp=.o)))

all: $(TEST_NAME) $(HIP_FILE).hsaco

$(TEST_NAME): $(OBJ_FILES) $(COMMON_OBJ_FILES)
	$(AOMP)/bin/clang++ $(OBJ_FILES) $(LFLAGS) -o $(TEST_NAME) -std=c++11

$(HIP_FILE).hsaco:
	$(AOMP)/bin/cloc.sh -mcpu $(MYGPU) $(HIP_FILE).cu -std=c++11

obj/%.o: %.cpp
	mkdir -p obj
	$(AOMP)/bin/clang++ $(CFLAGS) -o $@ $< -std=c++11

clean:
	rm -rf obj/*o *.hsaco *.ll $(TEST_NAME)

run:
	./$(TEST_NAME)
