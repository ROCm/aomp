include ../../Makefile.defs

TESTNAME     = hip-fopenmp
TESTSRC_MAIN = hip-fopenmp.cpp
TESTSRC_AUX  =
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
CLANG        ?= clang++ -x hip -D__HIP_PLATFORM_AMD__=1 $(LLIB) -lamdhip64 $(RPTH) -I$(AOMPHIP)/include
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules
