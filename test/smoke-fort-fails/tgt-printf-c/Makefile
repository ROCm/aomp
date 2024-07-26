include ../../Makefile.defs

TESTNAME     = main
TESTSRC_MAIN = main.f95
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang-new
OMP_BIN      = $(AOMP)/bin/$(FLANG)
#-ccc-print-phases
#"-\#\#\#"

#RUNENV      += LIBOMPTARGET_INFO=-1
RUNCMD       = ./${TESTNAME} |& tee /dev/fd/2 | ./chkit.sh

include ../Makefile.rules
main:
	${CC} -c util.c ${OMP_FLAGS}
	${OMP_BIN} main.f95 util.o ${OMP_FLAGS} -o main
