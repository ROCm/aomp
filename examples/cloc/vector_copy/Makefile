
TEST_NAME=vector_copy
CL_FILE=vector_copy

ifeq ($(AOMP),)
  AOMP = /opt/rocm/aomp
else
  $(info AOMP environment variable using $(AOMP))
endif

#LFLAGS= -g  -Wl,--unresolved-symbols=ignore-in-shared-libs
LFLAGS = 
CFLAGS =  -std=c++11

CXX_FILES := $(wildcard *.cpp)
OBJ_FILES := $(addprefix obj/, $(notdir $(CXX_FILES:.cpp=.o)))

all: $(TEST_NAME) $(CL_FILE).hsaco

$(TEST_NAME): $(OBJ_FILES) $(COMMON_OBJ_FILES)
	$(CXX) $(LFLAGS) $(OBJ_FILES) -L/opt/rocm/lib -lhsa-runtime64 -o $(TEST_NAME)

$(CL_FILE).hsaco :
	$(AOMP)/bin/cloc.sh -vv -ll $(CL_FILE).cl

obj/%.o: %.cpp
	mkdir -p obj
	$(CXX) -I/opt/rocm/include -c $(CFLAGS) -o $@ $< 

clean:
	rm -rf obj/*o *.hsaco $(TEST_NAME)


test:	
	./$(TEST_NAME)


