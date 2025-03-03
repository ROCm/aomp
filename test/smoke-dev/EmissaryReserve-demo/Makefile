TESTNAME        = EmissaryReserve-demo
TESTSRC         = EmissaryReserve-demo.cpp

MY_HOST_API_SRC = myHostAPI.cpp
MY_HOST_API_LIB = myHostAPI

AOMP           ?= $(HOME)/rocm/aomp
AOMP_GPU       ?= $(shell $(AOMP)/../../bin/rocm_agent_enumerator | grep -m 1 -E gfx[^0]{1}.{2})
CFLAGS          = -O3 -fopenmp --offload-arch=$(AOMP_GPU)
CLANG_DIR       = $(AOMP)/bin

$(TESTNAME): $(TESTSRC) lib$(MY_HOST_API_LIB).a
	$(CLANG_DIR)/clang++ $(CFLAGS) $(TESTSRC) -o $@ -L. -l$(MY_HOST_API_LIB) 

$(MY_HOST_API_LIB).o : $(MY_HOST_API_SRC)
	$(CLANG_DIR)/clang++ -c $(CFLAGS) $^ -o $@ 

lib$(MY_HOST_API_LIB).a : $(MY_HOST_API_LIB).o
	ar r $@ $^

run: $(TESTNAME)
	./$(TESTNAME)

clean:
	rm -f $(TESTNAME) obin *.i *.ii *.bc *.lk a.out-* *.ll *.s *.o *.cubin *.a
