
TEST_NAME=vector_copy
CL_FILE=vector_copy

ifeq ($(AOMP),)
  AOMP = /usr/lib/aomp
endif

AOMPHIP ?= $(AOMP)

CFLAGS = -std=c++11 -I$(AOMP)/include -I$(AOMP)/../include
LFLAGS = -L$(AOMP)/lib -L$(AOMP)/lib -lhsa-runtime64 -Wl,-rpath=$(AOMP)/lib
# find the current gpu, default to vega10 if unknown
AOMP_GPU ?= $(shell $(AOMPHIP)/bin/mygpu -d gfx900)

CXX_FILES := $(wildcard *.cpp)
OBJ_FILES := $(addprefix obj/, $(notdir $(CXX_FILES:.cpp=.o)))

all: $(TEST_NAME) $(CL_FILE).hsaco

obj/%.o: %.cpp
	mkdir -p obj
	$(CXX) -c $(CFLAGS) -o $@ $<

$(TEST_NAME): $(OBJ_FILES) $(COMMON_OBJ_FILES)
	$(CXX) $(OBJ_FILES) $(LFLAGS) -o $(TEST_NAME)

$(CL_FILE).hsaco :
	$(AOMPHIP)/bin/cloc.sh -cl12 -mcpu $(AOMP_GPU) $(CL_FILE).cl

clean:
	rm -rf obj/*o *.hsaco $(TEST_NAME)

run: $(TEST_NAME) $(CL_FILE).hsaco
	./$(TEST_NAME)


