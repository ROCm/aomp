include ../../Makefile.defs

TESTNAME	 = swdev-404405-main
TESTSRC_MAIN = swdev-404405-main.f90
# While these are not really the source files, these make up the prerequisites
TESTSRC_AUX  = fallocator.o ftimer.o timer.o
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG   ?= flang
CLANG		 = $(AOMP)/bin/clang++
CC           = $(AOMP)/bin/$(FLANG)
CFLAGS	     =
OMP_FLAGS    = -fopenmp $(TARGET)
LINK_FLAGS   = -lamdhip64 -lstdc++

RUNENV       = OMP_TARGET_OFFLOAD=MANDATORY

include ../Makefile.rules

fallocator.o: allocator.f90
	$(AOMP)/bin/$(FLANG) -c $^ -o $@

ftimer.o: timer.f90
	$(AOMP)/bin/$(FLANG) -c $^ -o $@

timer.o: timer.cpp
	$(CLANG) -c $^ -o $@
