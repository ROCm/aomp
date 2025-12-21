include ../../Makefile.defs

TESTNAME     = buildrun
TESTSRC_MAIN =
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
OMP_BIN      = $(AOMP)/bin/$(FLANG)
FC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"

include ../Makefile.rules

buildrun:
	${FC} ${OMP_FLAGS} -fPIC -c foo.F90 -o foo.o
	${FC} ${OMP_FLAGS} -fPIC -c main.F90 -o main.o
	${FC} -Wl,--disable-new-dtags -shared -Wl,-soname,libfoo.so -o libfoo.so foo.o -Wl,-rpath,"$$ORIGIN/../lib64"
	${FC} ${OMP_FLAGS} -Wl,--disable-new-dtags main.o -o test -L . -lfoo $(FLANG_GPU_LINK_FLAGS)
	LD_LIBRARY_PATH="." ./test
