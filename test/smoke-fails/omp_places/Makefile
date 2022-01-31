include ../../Makefile.defs

TESTNAME     = omp_places
TESTSRC_MAIN = omp_places.c
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

TARGET       =

CLANG        = clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)
#-ccc-print-phases
#"-\#\#\#"i

RUNENV += OMP_AFFINITY_FORMAT="Thread num: %n Affinity: %A" OMP_DISPLAY_AFFINITY=TRUE
RUNENV1 += $(RUNENV) OMP_PLACES="{0,1,2,3},{4,5,6,7},{8,9,10,11},{12,13,14,15}"
RUNENV2 += $(RUNENV) OMP_PLACES="{0:4},{4:4},{8:4},{12:4}"
RUNENV3 += $(RUNENV) OMP_PLACES="{0:4}:4:4"
RUNENV4 += $(RUNENV) OMP_PLACES="{0:16:2}"
RUNENV5 += $(RUNENV) OMP_PLACES=threads
RUNENV6 += $(RUNENV) OMP_PLACES="threads(7)"
RUNENV7 += $(RUNENV) OMP_PLACES="sockets(1)"
RUNENV8 += $(RUNENV) OMP_PLACES="sockets(2)"
RUNENV9 += $(RUNENV) OMP_PLACES="cores(3)"
RUNENV10 += $(RUNENV) OMP_PLACES="cores(7)"
RUNENV11 += $(RUNENV) OMP_PLACES="ll_caches(1)"
RUNENV12 += $(RUNENV) OMP_PLACES="ll_caches(2)"


include ../Makefile.rules
run:
	$(RUNENV) ./$(TESTNAME)
	$(RUNENV1) ./$(TESTNAME)
	$(RUNENV2) ./$(TESTNAME)
	$(RUNENV3) ./$(TESTNAME)
	$(RUNENV4) ./$(TESTNAME)
	$(RUNENV5) ./$(TESTNAME)
	$(RUNENV6) ./$(TESTNAME)
	$(RUNENV7) ./$(TESTNAME)
	$(RUNENV8) ./$(TESTNAME)
	$(RUNENV9) ./$(TESTNAME)
	$(RUNENV10) ./$(TESTNAME)
	$(RUNENV11) ./$(TESTNAME)
	$(RUNENV12) ./$(TESTNAME)
