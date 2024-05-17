include ../../Makefile.defs

TESTNAME     = flang-321838
TESTSRC_MAIN = flang-321838.f90
TESTSRC_AUX  =
TESTSRC_ALL  = $(TESTSRC_MAIN) $(TESTSRC_AUX)

FLANG        ?= flang
OMP_FLAGS    += -cpp -O3 -DPRIV_TS_RS -DM_TEAMS_N_THREADS
#OMP_FLAGS    += -cpp -DPRIV_TS_RS -DM_TEAMS
#OMP_FLAGS    += -cpp -O3 -DPRIV_TS_RS -DONE_TEAM
#OMP_FLAGS    += -cpp -O3 -DM_TEAMS_N_THREADS
#OMP_FLAGS    += -cpp -O3 -DM_TEAMS
#OMP_FLAGS    += -cpp -O3 -DONE_TEAM
OMP_BIN      = $(AOMP)/bin/$(FLANG)
CC           = $(OMP_BIN) $(VERBOSE)

include ../Makefile.rules
