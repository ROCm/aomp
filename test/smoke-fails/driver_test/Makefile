include ../../Makefile.defs

TESTNAME	= driver_test
# While these are not really the source files, these make up the prerequisites
TESTSRC_ALL = dec_arrayval.f95.o inc_arrayval.c.o main.c.o

FLANG       = $(AOMP)/bin/flang
CLANG		= $(AOMP)/bin/clang

COMMON      = -O3 -fopenmp $(TARGET) -ffast-math -c
COMMON     += -DPTRSIZE8 -DLONGINT8 -DUNDERSCORE -DGLOBAL_LONG_LONG
FFLAGS     	= $(COMMON) -r8 -i8 -cpp
CCFLAGS		= $(COMMON) -g -DPREFIX=jl_

CC          = $(FLANG)
CFLAGS	    = -O3 -fdisable-host-devmem -ffast-math -lm
OMP_FLAGS   = -fopenmp $(TARGET)

include ../Makefile.rules

dec_arrayval.f95.o: dec_arrayval.f95
	$(FLANG)  $(FFLAGS) $^ -o $@

inc_arrayval.c.o: inc_arrayval.c
	$(CLANG) $(CCFLAGS) $^ -o $@

main.c.o: main.c
	$(CLANG) $(CCFLAGS) $^ -o $@

.PHONY: clean
