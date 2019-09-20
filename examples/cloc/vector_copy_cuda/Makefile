
TEST_NAME=vector_copy
CU_FILE=vector_copy

ifeq ($(AOMP),)
  AOMP = /usr/lib/aomp
else
  $(info AOMP environment variable using $(AOMP))
endif

#LFLAGS= -g  -Wl,--unresolved-symbols=ignore-in-shared-libs
LFLAGS = 
CFLAGS =  -std=c++11
FUNOPTS = -vv -ll
FUNOPTS = 

CXX_FILES := $(wildcard *.cpp)
OBJ_FILES := $(addprefix obj/, $(notdir $(CXX_FILES:.cpp=.o)))

all: $(TEST_NAME) $(CU_FILE).hsaco

$(TEST_NAME): $(OBJ_FILES) $(COMMON_OBJ_FILES)
	$(CXX) $(LFLAGS) $(OBJ_FILES) -L$(AOMP)/hsa/lib -lhsa-runtime64 -o $(TEST_NAME) -Wl,-rpath=$(AOMP)/hsa/lib

$(CU_FILE).hsaco :
	$(AOMP)/bin/cloc.sh $(FUNOPTS)  $(CU_FILE).cu

obj/%.o: %.cpp
	mkdir -p obj
	$(CXX) -I$(AOMP)/hsa/include -c $(CFLAGS) -o $@ $<

clean:
	rm -rf obj/*o *.hsaco *.ll $(TEST_NAME)

run:
	./$(TEST_NAME)
