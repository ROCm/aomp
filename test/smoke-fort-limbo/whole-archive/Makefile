#NOOPT        = 1
#NOOMP        = 1
#NOLINK       = 1
include ../../Makefile.defs

TESTNAME     = test
TESTSRC_MAIN = main.f90
TESTSRC_AUX  = vadd.f90
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
CFLAGS       = -c
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules

#LDFLAGS=-Wl,--whole-archive -Wl,--allow-multiple-definition -Wl,--no-nvptx-whole-archive
LDFLAGS=-Wl,--whole-archive -Wl,--allow-multiple-definition

test:
	${OMP_BIN} -c ${OMP_FLAGS} vadd.f90
	rm -f vadd.a
	ar r vadd.a vadd.o
	${OMP_BIN} -c main.f90
	${OMP_BIN} -o test main.o ${LDFLAGS} ${OMP_FLAGS} vadd.a

clean::
	rm -f vadd.a
