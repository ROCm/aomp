
TEST_NAME=vector_copy
CU_FILE=vector_copy

ifeq ($(AOMP),)
  AOMP = /opt/rocm/aomp
else
  $(info AOMP environment variable using $(AOMP))
endif

#LFLAGS= -g  -Wl,--unresolved-symbols=ignore-in-shared-libs
LFLAGS = 
CFLAGS =  -std=c++11 -D__HIP_PLATFORM_HCC__
FUNOPTS = -vv -ll
FUNOPTS = 

CXX_FILES := $(wildcard *.cpp)
OBJ_FILES := $(addprefix obj/, $(notdir $(CXX_FILES:.cpp=.o)))

all: $(TEST_NAME) $(CU_FILE).hsaco

$(TEST_NAME): $(OBJ_FILES) $(COMMON_OBJ_FILES)
	$(AOMP)/bin/clang++ $(OBJ_FILES) -L$(AOMP)/lib -lhip_hcc -o $(TEST_NAME) $(LFLAGS)
#	$(CXX) $(OBJ_FILES) -L$(AOMP)/lib -lhip_hcc -o $(TEST_NAME) $(LFLAGS)

$(CU_FILE).hsaco :
	$(AOMP)/bin/cloc.sh $(FUNOPTS)  $(CU_FILE).cu

obj/%.o: %.cpp
	mkdir -p obj
	$(AOMP)/bin/clang++ -I$(AOMP)/include -c $(CFLAGS) -o $@ $< 
#	$(CXX) -I/opt/rocm/include -c $(CFLAGS) -o $@ $< 

clean:
	rm -rf obj/*o *.hsaco *.ll $(TEST_NAME)


test:	
	./$(TEST_NAME)


