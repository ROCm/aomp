
include ../../Makefile.defs

TESTNAME =hip-gpu-burn
FILETYPE =cpp

ifeq ($(TARGETS),)
TARGETS =--offload-arch=$(AOMP_GPU)
endif

CC       =$(AOMP)/bin/clang++ -std=c++11
CFLAGS   =-x hip -O3 $(AOMP_CPUTARGET) 
# Add cudart only if we have an Nvidia sm_ target
ifeq (sm_,$(findstring sm_,$(TARGETS)))
  LFLAGS +=-L$(CUDA)/targets/$(UNAMEP)-linux/lib -lcudart -Wl,-rpath,$(CUDA)/targets/$(UNAMEP)-linux/lib
  CFLAGS +=-x cuda -I$(CUDA)/include
else
  HIPLIBS = -L $(AOMPHIP)/hip -L $(AOMPHIP)/lib
  VERS = $(shell $(AOMP)/bin/clang --version | grep -oP '(?<=clang version )[0-9.]+')
  ifeq ($(shell expr $(VERS) \>= 12.0), 1)
    RPTH = -Wl,-rpath,$(AOMPHIP)/lib
  endif
  CFLAGS += -x hip $(HIPLIBS) -lamdhip64 $(RPTH)
endif

# ----- Demo compile and link in one step, no object code saved
$(TESTNAME): $(TESTNAME).$(FILETYPE)
	$(CC) $(CFLAGS) $(TARGETS) $(LFLAGS) $^ -o $@

run: $(TESTNAME)
	./$(TESTNAME)

#  ----   Demo compile and link in two steps, object saved
$(TESTNAME).o: $(TESTNAME).$(FILETYPE)
	$(CC) -c $(CFLAGS) $(TARGETS) $^

obin:	$(TESTNAME).o 
	$(CC) $(LFLAGS) $^ -o obin

run_obin: obin 
	./obin

#  ----   Demo compile to intermediates LLVMIR or assembly
$(TESTNAME).ll: $(TESTNAME).$(FILETYPE)
	$(CC) -c -S -emit-llvm $(CFLAGS) $(TARGETS) $^

$(TESTNAME).s: $(TESTNAME).$(FILETYPE)
	$(CC)  -c -S $(CFLAGS) $(TARGETS)  $^
