
TEST_NAME=vector_copy
CL_FILE=vector_copy

ifeq ($(AOMP),)
  AOMP = /usr/lib/aomp
endif

CFLAGS = -std=c++11 -I$(AOMP)/include
LFLAGS = -L$(AOMP)/lib -lhsa-runtime64 -Wl,-rpath=$(AOMP)/hsa/lib
# find the current gpu, default to vega10 if unknown
MYGPU = $(shell $(AOMP)/bin/mygpu -d gfx900)

CXX_FILES := $(wildcard *.cpp)
OBJ_FILES := $(addprefix obj/, $(notdir $(CXX_FILES:.cpp=.o)))

all: $(TEST_NAME) $(CL_FILE).hsaco

obj/%.o: %.cpp
	mkdir -p obj
	$(CXX) -c $(CFLAGS) -o $@ $<

$(TEST_NAME): $(OBJ_FILES) $(COMMON_OBJ_FILES)
	$(CXX) $(OBJ_FILES) $(LFLAGS) -o $(TEST_NAME)

$(CL_FILE).hsaco :
	$(AOMP)/bin/cloc.sh -cl12 -mcpu $(MYGPU) $(CL_FILE).cl

clean:
	rm -rf obj/*o *.hsaco $(TEST_NAME)

run: $(TEST_NAME) $(CL_FILE).hsaco
	./$(TEST_NAME)


