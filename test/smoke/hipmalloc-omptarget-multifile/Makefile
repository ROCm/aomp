include ../../Makefile.defs

TESTNAME     = main
TESTSRC_MAIN = main.cpp
TESTSRC_AUX  = support.o hipMallocOmpTarget1.o hipMallocOmpTarget2.o
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)
VERS = $(shell $(AOMP)/bin/clang --version | grep -oP '(?<=clang version )[0-9.]+')
ifeq ($(shell expr $(VERS) \>= 12.0), 1)
ifeq ($(AOMP_SANITIZER),1)
  RPTH = -Wl,-rpath,$(AOMPHIP)/lib/asan
  LLIB = -L$(AOMPHIP)/lib/asan
else
  RPTH = -Wl,-rpath,$(AOMPHIP)/lib
  LLIB = -L$(AOMPHIP)/lib
endif
endif
CLANG        ?= clang++ -D__HIP_PLATFORM_AMD__=1 -I$(AOMPHIP)/include -fopenmp
LINK_FLAGS   = $(LLIB) -lamdhip64 $(RPTH)
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules

support.o: support.cpp
	$(CC) -c $^ -o $@
hipMallocOmpTarget1.o: hipMallocOmpTarget1.cpp
	$(CC) $(TARGET) -c $^ -o $@
hipMallocOmpTarget2.o: hipMallocOmpTarget2.cpp
	$(CC) $(TARGET) -c $^ -o $@
