
TEST_NAME=vector_copy
HIP_FILE=vector_copy

ifeq ($(AOMP),)
  AOMP = /usr/lib/aomp
endif

CFLAGS = -c -std=c++11 -D__HIP_PLATFORM_HCC__
# Set the rpath to pick up AOMP libs in case rocm was installed.
LFLAGS = -L$(AOMP)/lib -lhip_hcc -Wl,-rpath,$(AOMP)/lib
# find the current gpu, default to vega10 if unknown
MYGPU = $(shell $(AOMP)/bin/mygpu -d gfx900)

CXX_FILES := $(wildcard *.cpp)
OBJ_FILES := $(addprefix obj/, $(notdir $(CXX_FILES:.cpp=.o)))

all: $(TEST_NAME) $(HIP_FILE).hsaco

$(TEST_NAME): $(OBJ_FILES) $(COMMON_OBJ_FILES)
	$(AOMP)/bin/clang++ $(OBJ_FILES) $(LFLAGS) -o $(TEST_NAME)

$(HIP_FILE).hsaco:
	$(AOMP)/bin/cloc.sh -mcpu $(MYGPU) $(HIP_FILE).cu

obj/%.o: %.cpp
	mkdir -p obj
	$(AOMP)/bin/clang++ $(CFLAGS) -o $@ $<

clean:
	rm -rf obj/*o *.hsaco *.ll $(TEST_NAME)

run:
	./$(TEST_NAME)
